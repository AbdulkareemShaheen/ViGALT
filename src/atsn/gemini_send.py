#!/usr/bin/env python3
"""
Gemini Web Automation - Proof of Concept

Automates gemini.google.com (not the API) to send a prompt with an image attachment.

First run:
  1. python -m venv .venv
  2. .venv\\Scripts\\activate          (Windows)
  3. pip install -r requirements.txt
  4. playwright install chromium
  5. pip install -e .
  6. python -m atsn.gemini_send

On first launch the browser opens with a fresh profile. If Google asks you to sign in,
complete login in the browser window, return to the terminal, and press Enter.
Your session is saved in .browser_profile/ for future runs.

Usage:
  python -m atsn.gemini_send
  python -m atsn.gemini_send --questions questions.json --images-dir data/images
  python -m atsn.gemini_send --single --prompt "what is this image" --image data/images/1_clothing.jpg
  python -m atsn.gemini_send --product data/products/1_clothing.json --prompt-file prompts/classifier.txt
  python -m atsn.gemini_send --keep-open
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from playwright.sync_api import (
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

GEMINI_URL = "https://gemini.google.com/app"
DEFAULT_PROMPT = "what is this image"
DEFAULT_IMAGE = "data/images/1_clothing.jpg"
DEFAULT_QUESTIONS_FILE = "questions.json"
DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_PROMPT_FILE = "prompts/classifier.txt"
DEFAULT_PRODUCT_FILE = "data/products/1_clothing.json"
DOWNLOADS_DIR = "downloads"
PROFILE_DIR = ".browser_profile"
OUTPUT_DIR = "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@dataclass
class QuestionJob:
    index: int
    question: str
    image_path: Path


@dataclass
class ProductJob:
    index: int
    product_file: Path
    image_url: str
    image_path: Path
    prompt: str
    result_file: Path

# Centralized selectors — update here if Gemini UI changes.
UPLOAD_TOOLS_MENU_SELECTORS = [
    'button[aria-label="Upload & tools"]',
    'button[aria-label*="Upload & tools"]',
    'button:has-text("Upload & tools")',
]

UPLOAD_FILES_MENU_SELECTORS = [
    '[role="menuitem"]:has-text("Upload files")',
    '[role="menuitem"]:has-text("Add files")',
    'button:has-text("Upload files")',
    'button:has-text("Add files")',
    '[aria-label*="Upload file"]',
]

CHAT_INPUT_SELECTORS = [
    'div[contenteditable="true"][role="textbox"]',
    'rich-textarea div[contenteditable="true"]',
    '.ql-editor[contenteditable="true"]',
    'textarea[aria-label*="Enter a prompt"]',
    'textarea[placeholder*="Enter"]',
]

SEND_BUTTON_SELECTORS = [
    'button[aria-label*="Send"]',
    'button[aria-label*="send"]',
    'button.send-button',
    'button[mattooltip*="Send"]',
]

ASSISTANT_MESSAGE_SELECTORS = [
    ".model-response-text",
    "message-content",
    '[data-message-author-role="model"]',
    ".response-container",
    "model-response",
]

STOP_BUTTON_SELECTORS = [
    'button[aria-label*="Stop"]',
    'button[aria-label*="stop"]',
]

ATTACHMENT_SELECTORS = [
    "img[src*='blob:']",
    "img[src*='googleusercontent']",
    ".file-preview",
    ".attachment-preview",
    "images-files-uploader img",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send prompts with images to Gemini via the web UI."
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS_FILE,
        help=f"JSON file with question objects (default: {DEFAULT_QUESTIONS_FILE})",
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Folder containing images, matched by sort order (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--product",
        default=None,
        help="Product JSON file with main_image URL (e.g. data/products/1_clothing.json).",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt template for --product mode (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Send one prompt/image instead of reading questions.json.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt text for --single mode (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Image path for --single mode (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the browser open after completion until you press Enter.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120_000,
        help="Response timeout in milliseconds (default: 120000).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between batch requests (default: 2.0).",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def resolve_image_path(image_arg: str) -> Path:
    image_path = resolve_path(image_arg)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def load_questions(questions_file: Path) -> list[str]:
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    data = json.loads(questions_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{questions_file} must contain a JSON array.")

    questions: list[str] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(
                f"Item {i} in {questions_file} must be an object with a 'question' field."
            )
        question = str(item["question"]).strip()
        if not question:
            raise ValueError(f"Item {i} in {questions_file} has an empty question.")
        questions.append(question)
    return questions


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return images


def image_filename_from_url(image_url: str) -> str:
    path = urlparse(image_url).path
    name = Path(path).name
    if name and "." in name:
        return name
    return "downloaded_image.jpg"


def download_image(image_url: str, downloads_dir: Path) -> Path:
    downloads_dir.mkdir(parents=True, exist_ok=True)
    filename = image_filename_from_url(image_url)
    image_path = downloads_dir / filename

    print(f"Downloading image: {image_url}")
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    image_path.write_bytes(response.content)
    print(f"Saved image to: {image_path}")
    return image_path.resolve()


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").rstrip()


def prepare_product(product_file: Path, prompt_file: Path, downloads_dir: Path) -> ProductJob:
    if not product_file.exists():
        raise FileNotFoundError(f"Product file not found: {product_file}")

    data = json.loads(product_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{product_file} must contain a JSON object.")

    image_url = data.get("main_image")
    if not image_url:
        raise ValueError(f"{product_file} is missing a 'main_image' field.")

    image_url = str(image_url).strip()
    if not image_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid main_image URL in {product_file}: {image_url!r}")

    del data["main_image"]
    product_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Removed main_image from: {product_file}")

    image_path = download_image(image_url, downloads_dir)
    prompt_template = load_prompt_template(prompt_file)
    product_json = json.dumps(data, indent=2, ensure_ascii=False)
    prompt = f"{prompt_template}\n{product_json}"

    result_file = product_file.with_name(f"{product_file.stem}_gemini_response.json")
    return ProductJob(
        index=1,
        product_file=product_file,
        image_url=image_url,
        image_path=image_path,
        prompt=prompt,
        result_file=result_file,
    )


def build_product_jobs(args: argparse.Namespace) -> list[ProductJob]:
    product_file = resolve_path(args.product or DEFAULT_PRODUCT_FILE)
    prompt_file = resolve_path(args.prompt_file)
    downloads_dir = project_root() / DOWNLOADS_DIR
    return [prepare_product(product_file, prompt_file, downloads_dir)]


def build_jobs(args: argparse.Namespace) -> list[QuestionJob]:
    if args.single:
        return [
            QuestionJob(
                index=1,
                question=args.prompt,
                image_path=resolve_image_path(args.image),
            )
        ]

    questions_file = resolve_path(args.questions)
    images_dir = resolve_path(args.images_dir)
    questions = load_questions(questions_file)
    images = list_images(images_dir)

    if len(questions) != len(images):
        raise ValueError(
            f"Question/image count mismatch: {len(questions)} question(s) in "
            f"{questions_file.name}, but {len(images)} image(s) in {images_dir.name}.\n"
            f"Images found: {', '.join(img.name for img in images)}"
        )

    return [
        QuestionJob(index=i + 1, question=question, image_path=image)
        for i, (question, image) in enumerate(zip(questions, images))
    ]


def launch_context(playwright: Playwright, profile_dir: Path) -> BrowserContext:
    profile_dir.mkdir(parents=True, exist_ok=True)
    launch_kwargs = {
        "user_data_dir": str(profile_dir),
        "headless": False,
        "viewport": {"width": 1280, "height": 900},
        "accept_downloads": True,
        "args": ["--disable-blink-features=AutomationControlled"],
    }
    try:
        return playwright.chromium.launch_persistent_context(channel="chrome", **launch_kwargs)
    except Exception:
        return playwright.chromium.launch_persistent_context(**launch_kwargs)


def first_visible(page: Page, selectors: list[str], timeout: int = 15_000):
    deadline = time.time() + (timeout / 1000)
    while time.time() < deadline:
        for selector in selectors:
            locator = page.locator(selector).first
            try:
                if locator.count() > 0 and locator.is_visible():
                    return locator
            except Exception:
                continue
        page.wait_for_timeout(300)
    return None


def wait_for_gemini_ready(page: Page) -> None:
    print("Waiting for Gemini chat UI...")
    locator = first_visible(page, CHAT_INPUT_SELECTORS, timeout=60_000)
    if locator is None:
        raise RuntimeError(
            "Could not find the Gemini chat input. "
            "Make sure you are logged in and on gemini.google.com/app."
        )


def safe_input(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        print("(Non-interactive terminal: skipping wait for Enter.)")


def needs_manual_login(page: Page) -> bool:
    url = page.url.lower()
    if "accounts.google.com" in url:
        return True
    if "signin" in url or "servicelogin" in url:
        return True
    if page.locator('input[type="email"]').count() > 0:
        return True

    sign_in_btn = page.get_by_role("button", name=re.compile(r"^Sign in$", re.I))
    try:
        if sign_in_btn.count() > 0 and sign_in_btn.first.is_visible():
            return True
    except Exception:
        pass

    return False


def start_new_chat(page: Page) -> None:
    new_chat = page.get_by_role("button", name=re.compile(r"New chat", re.I)).first
    if new_chat.count() > 0 and new_chat.is_visible():
        new_chat.click()
    else:
        page.goto(GEMINI_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(2000)
    wait_for_gemini_ready(page)


def pause_for_login(page: Page) -> None:
    if not needs_manual_login(page):
        wait_for_gemini_ready(page)
        return

    print("\n" + "=" * 60)
    print("Google login required.")
    print("1. Sign in to your Google account in the browser window.")
    print("2. Wait until Gemini chat loads.")
    print("3. Return here and press Enter to continue.")
    print("=" * 60 + "\n")
    safe_input("Press Enter after you have signed in... ")

    if "gemini.google.com" not in page.url:
        page.goto(GEMINI_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)

    wait_for_gemini_ready(page)


def open_upload_tools_menu(page: Page) -> None:
    menu_btn = page.get_by_role("button", name=re.compile(r"Upload.*tools", re.I)).first
    if menu_btn.count() == 0 or not menu_btn.is_visible():
        menu_btn = first_visible(page, UPLOAD_TOOLS_MENU_SELECTORS, timeout=20_000)
    if menu_btn is None:
        raise RuntimeError(
            "Could not find the 'Upload & tools' button. "
            "Gemini UI may have changed — update UPLOAD_TOOLS_MENU_SELECTORS."
        )
    menu_btn.click()
    page.wait_for_timeout(700)


def click_upload_files_menu_item(page: Page) -> None:
    upload_item = page.get_by_role(
        "menuitem", name=re.compile(r"upload.*files?", re.I)
    ).first
    if upload_item.count() == 0 or not upload_item.is_visible():
        upload_item = first_visible(page, UPLOAD_FILES_MENU_SELECTORS, timeout=10_000)
    if upload_item is None:
        sign_in_tools = page.get_by_role(
            "button", name=re.compile(r"Sign in to try tools", re.I)
        ).first
        if sign_in_tools.count() > 0 and sign_in_tools.is_visible():
            raise RuntimeError(
                "Upload requires a signed-in Google account. "
                "Run again, sign in when prompted, then retry."
            )
        raise RuntimeError(
            "Could not find 'Upload files' in the tools menu. "
            "Gemini UI may have changed — update UPLOAD_FILES_MENU_SELECTORS."
        )
    upload_item.click()


def upload_image(page: Page, image_path: Path) -> None:
    print(f"Uploading image: {image_path.name}")

    for attempt in range(2):
        try:
            with page.expect_file_chooser(timeout=10_000) as fc_info:
                open_upload_tools_menu(page)
                click_upload_files_menu_item(page)
            file_chooser = fc_info.value
            file_chooser.set_files(str(image_path))
            break
        except PlaywrightTimeoutError:
            if attempt == 1:
                # Fallback: hidden file input may already exist in DOM.
                file_input = page.locator('input[type="file"]').first
                if file_input.count() > 0:
                    file_input.set_input_files(str(image_path))
                    break
                raise RuntimeError(
                    "File chooser did not appear. "
                    "Try clicking Upload manually once, then rerun."
                ) from None
            page.wait_for_timeout(1000)

    # Wait for attachment preview to appear.
    attached = False
    for _ in range(30):
        for selector in ATTACHMENT_SELECTORS:
            if page.locator(selector).count() > 0:
                attached = True
                break
        if attached:
            break
        page.wait_for_timeout(500)

    if attached:
        print("Image attached successfully.")
    else:
        print("Warning: attachment preview not detected, continuing anyway...")


def send_prompt(page: Page, prompt: str) -> None:
    print(f"Sending prompt: {prompt!r}")

    chat_input = page.get_by_role(
        "textbox", name=re.compile(r"Enter a prompt", re.I)
    ).first
    if chat_input.count() == 0 or not chat_input.is_visible():
        chat_input = first_visible(page, CHAT_INPUT_SELECTORS, timeout=15_000)
    if chat_input is None:
        raise RuntimeError("Could not find chat input to type the prompt.")

    chat_input.click()
    page.wait_for_timeout(200)
    chat_input.fill(prompt)
    page.wait_for_timeout(300)
    page.keyboard.press("Enter")


def is_generating(page: Page) -> bool:
    stop_btn = page.get_by_role("button", name=re.compile(r"Stop", re.I)).first
    try:
        if stop_btn.count() > 0 and stop_btn.is_visible():
            return True
    except Exception:
        pass
    for selector in STOP_BUTTON_SELECTORS:
        locator = page.locator(selector).first
        try:
            if locator.count() > 0 and locator.is_visible():
                return True
        except Exception:
            continue
    return False


def extract_latest_response(page: Page) -> str:
    for selector in ASSISTANT_MESSAGE_SELECTORS:
        messages = page.locator(selector)
        count = messages.count()
        if count > 0:
            text = messages.nth(count - 1).inner_text(timeout=5_000).strip()
            if text:
                return text

    # Broad fallback: last substantial text block in the conversation area.
    candidates = page.locator("main").locator("p, div, span").all_inner_texts()
    for text in reversed(candidates):
        cleaned = text.strip()
        if len(cleaned) > 20:
            return cleaned

    return ""


def wait_for_response(page: Page, timeout_ms: int) -> str:
    print("Waiting for Gemini response...")

    # Give Gemini a moment to start generating.
    page.wait_for_timeout(1500)

    deadline = time.time() + (timeout_ms / 1000)
    last_text = ""
    stable_count = 0

    while time.time() < deadline:
        generating = is_generating(page)

        current_text = extract_latest_response(page)
        if current_text and current_text == last_text and not generating:
            stable_count += 1
            if stable_count >= 3:
                return current_text
        else:
            stable_count = 0
            last_text = current_text

        page.wait_for_timeout(1000)

    if last_text:
        print("Warning: timed out waiting for response to stabilize; returning partial text.")
        return last_text

    raise RuntimeError("No response received within the timeout period.")


def save_screenshot(page: Page, output_dir: Path, filename: str = "last_response.png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / filename
    page.screenshot(path=str(screenshot_path), full_page=True)
    return screenshot_path


def save_results(output_dir: Path, results: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "responses.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results_path


def save_product_response(result_file: Path, image_url: str, response: str) -> Path:
    payload = {
        "image_url": image_url,
        "response": response,
    }
    result_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result_file


def process_product_job(
    page: Page, job: ProductJob, timeout_ms: int, output_dir: Path
) -> dict:
    print("\n" + "-" * 60)
    print(f"Product: {job.product_file.name}")
    print(f"Image:   {job.image_path.name}")
    print(f"URL:     {job.image_url}")
    print("-" * 60)

    start_new_chat(page)
    upload_image(page, job.image_path)
    send_prompt(page, job.prompt)
    response = wait_for_response(page, timeout_ms)

    screenshot_name = f"{job.product_file.stem}_response.png"
    save_screenshot(page, output_dir, screenshot_name)
    result_path = save_product_response(job.result_file, job.image_url, response)

    print("\nResponse:")
    print(response)
    print(f"\nSaved to: {result_path}")

    return {
        "product_file": str(job.product_file),
        "image_url": job.image_url,
        "response": response,
        "result_file": str(result_path),
        "screenshot": screenshot_name,
    }


def process_job(page: Page, job: QuestionJob, timeout_ms: int, output_dir: Path) -> dict:
    print("\n" + "-" * 60)
    print(f"Job {job.index}")
    print(f"Image:    {job.image_path.name}")
    print(f"Question: {job.question}")
    print("-" * 60)

    start_new_chat(page)
    upload_image(page, job.image_path)
    send_prompt(page, job.question)
    response = wait_for_response(page, timeout_ms)

    screenshot_name = f"response_{job.index:02d}.png"
    text_name = f"response_{job.index:02d}.txt"
    save_screenshot(page, output_dir, screenshot_name)
    (output_dir / text_name).write_text(response, encoding="utf-8")

    print("\nResponse:")
    print(response)

    return {
        "index": job.index,
        "question": job.question,
        "image": job.image_path.name,
        "response": response,
        "screenshot": screenshot_name,
        "text_file": text_name,
    }


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_product(args: argparse.Namespace) -> int:
    configure_stdout()

    root = project_root()
    profile_dir = root / PROFILE_DIR
    output_dir = root / OUTPUT_DIR
    jobs = build_product_jobs(args)

    print("=" * 60)
    print("Gemini Product Automation")
    print("=" * 60)
    print(f"Jobs:    {len(jobs)}")
    print(f"Profile: {profile_dir}")
    for job in jobs:
        print(f"  {job.index}. {job.product_file.name} -> {job.image_path.name}")
    print()

    with sync_playwright() as playwright:
        context = launch_context(playwright, profile_dir)
        page = context.pages[0] if context.pages else context.new_page()
        results: list[dict] = []

        try:
            print(f"Opening {GEMINI_URL} ...")
            page.goto(GEMINI_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            pause_for_login(page)

            for job in jobs:
                result = process_product_job(page, job, args.timeout, output_dir)
                results.append(result)
                if job.index < len(jobs):
                    page.wait_for_timeout(int(args.delay * 1000))

            save_screenshot(page, output_dir)

            print("\n" + "=" * 60)
            print("PRODUCT RESPONSE SAVED")
            print("=" * 60)
            for result in results:
                print(f"  {result['result_file']}")

            if args.keep_open:
                safe_input("\nPress Enter to close the browser...")
            else:
                print("\nBrowser will close in 5 seconds (use --keep-open to keep it)...")
                page.wait_for_timeout(5000)

            return 0

        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            try:
                err_shot = save_screenshot(page, output_dir)
                print(f"Error screenshot saved: {err_shot}", file=sys.stderr)
            except Exception:
                pass
            if args.keep_open:
                safe_input("\nPress Enter to close the browser...")
            return 1

        finally:
            context.close()


def run_questions(args: argparse.Namespace) -> int:
    configure_stdout()

    root = project_root()
    profile_dir = root / PROFILE_DIR
    output_dir = root / OUTPUT_DIR
    jobs = build_jobs(args)

    print("=" * 60)
    print("Gemini Web Automation")
    print("=" * 60)
    print(f"Jobs:    {len(jobs)}")
    print(f"Profile: {profile_dir}")
    for job in jobs:
        print(f"  {job.index}. {job.image_path.name} -> {job.question}")
    print()

    with sync_playwright() as playwright:
        context = launch_context(playwright, profile_dir)
        page = context.pages[0] if context.pages else context.new_page()
        results: list[dict] = []

        try:
            print(f"Opening {GEMINI_URL} ...")
            page.goto(GEMINI_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            pause_for_login(page)

            for job in jobs:
                result = process_job(page, job, args.timeout, output_dir)
                results.append(result)
                if job.index < len(jobs):
                    page.wait_for_timeout(int(args.delay * 1000))

            results_path = save_results(output_dir, results)
            save_screenshot(page, output_dir)

            print("\n" + "=" * 60)
            print("ALL RESPONSES SAVED")
            print("=" * 60)
            print(f"Summary: {results_path}")
            for result in results:
                print(f"  Job {result['index']}: {result['text_file']}")

            if args.keep_open:
                safe_input("\nPress Enter to close the browser...")
            else:
                print("\nBrowser will close in 5 seconds (use --keep-open to keep it)...")
                page.wait_for_timeout(5000)

            return 0

        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            if results:
                results_path = save_results(output_dir, results)
                print(f"Partial results saved: {results_path}", file=sys.stderr)
            try:
                err_shot = save_screenshot(page, output_dir)
                print(f"Error screenshot saved: {err_shot}", file=sys.stderr)
            except Exception:
                pass
            if args.keep_open:
                safe_input("\nPress Enter to close the browser...")
            return 1

        finally:
            context.close()


def run(args: argparse.Namespace) -> int:
    if args.product is not None:
        return run_product(args)
    return run_questions(args)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
