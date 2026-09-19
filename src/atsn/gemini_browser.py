"""Playwright helpers for automating gemini.google.com."""

from __future__ import annotations

import re
import time
from pathlib import Path

from playwright.sync_api import (
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

GEMINI_URL = "https://gemini.google.com/app"

MODEL_MAP: dict[str, re.Pattern[str]] = {
    "gemini-3.5-flash-lite": re.compile(r"3\.5\s*flash[- ]lite", re.I),
    "gemini-3.1-pro-preview": re.compile(r"3\.1\s*pro", re.I),
    # UI may label this 3.6/3.8 Flash when 3.7 is unavailable.
    "gemini-3.7-flash": re.compile(r"3\.[678]\s*flash(?!.*lite)", re.I),
}

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

MODEL_MENU_ITEM_SELECTORS = [
    '[role="menuitem"]',
    '[role="option"]',
    '[role="listbox"] [role="option"]',
    "mat-option",
    ".mat-mdc-menu-item",
    ".model-picker-item",
]

RESPONSE_NOISE_PATTERNS = [
    re.compile(r"^opens in a new window\.?$", re.I),
    re.compile(r"^(copy|share|more|edit|retry)$", re.I),
    re.compile(r"^gemini$", re.I),
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


def _normalize_menu_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _model_menu_score(text: str, pattern: re.Pattern[str]) -> int:
    normalized = _normalize_menu_text(text)
    if not pattern.search(normalized):
        return -1
    # Prefer versioned labels over generic "Flash".
    if re.search(r"3\.\d", normalized):
        return 100 + len(normalized)
    return 10 + len(normalized)


def _is_model_selector_label(text: str) -> bool:
    normalized = _normalize_menu_text(text)
    return bool(
        re.match(
            r"^(Flash(-Lite)?|Flash\s*Lite|Pro|3\.\d+\s+(Pro|Flash(-Lite)?))$",
            normalized,
            re.I,
        )
    )


def _open_model_selector(page: Page) -> bool:
    """Open the model dropdown attached to the prompt input bar."""
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(400)

    deadline = time.time() + 15
    while time.time() < deadline:
        buttons = page.get_by_role("button")
        for i in range(buttons.count()):
            btn = buttons.nth(i)
            try:
                if not btn.is_visible():
                    continue
                text = btn.inner_text(timeout=500).strip()
                if _is_model_selector_label(text):
                    btn.click()
                    page.wait_for_timeout(900)
                    return True
            except Exception:
                continue

        for candidate in [
            page.locator('[aria-label*="model" i]').last,
            page.locator("button").filter(
                has_text=re.compile(r"^(Flash(-Lite)?|Pro)$", re.I)
            ).last,
        ]:
            try:
                if candidate.count() > 0 and candidate.is_visible():
                    candidate.click()
                    page.wait_for_timeout(900)
                    return True
            except Exception:
                continue

        page.wait_for_timeout(300)
    return False


def _collect_model_menu_items(page: Page) -> list[tuple[str, object]]:
    collected: list[tuple[str, object]] = []
    seen: set[str] = set()

    selectors = MODEL_MENU_ITEM_SELECTORS + ["button"]
    for selector in selectors:
        items = page.locator(selector)
        for i in range(items.count()):
            item = items.nth(i)
            try:
                if not item.is_visible():
                    continue
                text = item.inner_text(timeout=1_000).strip()
                if not text or len(text) > 120:
                    continue
                normalized = _normalize_menu_text(text)
                if normalized in seen:
                    continue
                if not re.search(r"flash|pro|thinking|gemini", normalized, re.I):
                    continue
                seen.add(normalized)
                collected.append((text, item))
            except Exception:
                continue
    return collected


def _find_model_menu_item(page: Page, pattern: re.Pattern[str]):
    best_item = None
    best_score = -1

    for text, item in _collect_model_menu_items(page):
        score = _model_menu_score(text, pattern)
        if score > best_score:
            best_score = score
            best_item = item

    return best_item


def select_model(page: Page, model_key: str) -> None:
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model key: {model_key!r}. Known: {list(MODEL_MAP)}")

    pattern = MODEL_MAP[model_key]
    print(f"Selecting model: {model_key}")

    if not _open_model_selector(page):
        raise RuntimeError(
            "Could not open the Gemini model selector. "
            "Gemini UI may have changed — update MODEL_SELECTOR_SELECTORS."
        )

    menu_item = _find_model_menu_item(page, pattern)
    if menu_item is None:
        raise RuntimeError(
            f"Could not find model matching {model_key!r} in the model picker. "
            "Check MODEL_MAP regex patterns against current Gemini UI labels."
        )

    menu_item.click()
    page.wait_for_timeout(1500)
    wait_for_gemini_ready(page)
    print(f"Model selected: {model_key}")


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
                file_input = page.locator('input[type="file"]').first
                if file_input.count() > 0:
                    file_input.set_input_files(str(image_path))
                    break
                raise RuntimeError(
                    "File chooser did not appear. "
                    "Try clicking Upload manually once, then rerun."
                ) from None
            page.wait_for_timeout(1000)

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
    preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
    print(f"Sending prompt ({len(prompt)} chars): {preview!r}")

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


def is_noise_response(text: str, *, min_length: int = 15) -> bool:
    cleaned = text.strip()
    if len(cleaned) < min_length:
        return True
    for pattern in RESPONSE_NOISE_PATTERNS:
        if pattern.search(cleaned):
            return True
    return False


def _strip_gemini_prefix(text: str) -> str:
    return re.sub(r"^Gemini said\s*\n+", "", text.strip(), flags=re.I).strip()


def looks_like_json_response(text: str) -> bool:
    cleaned = _strip_gemini_prefix(text)
    return cleaned.startswith("{") or '"stage"' in cleaned


def json_response_complete(text: str) -> bool:
    """Return True when a streamed JSON object appears fully closed."""
    cleaned = _strip_gemini_prefix(text)
    start = cleaned.find("{")
    if start == -1:
        return True

    depth = 0
    in_string = False
    escape = False
    for ch in cleaned[start:]:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def _element_text(locator) -> str:
    for getter in (
        lambda: locator.inner_text(timeout=5_000).strip(),
        lambda: str(locator.evaluate("el => (el.textContent || el.innerText || '').trim()")).strip(),
    ):
        try:
            text = getter()
            if text:
                return text
        except Exception:
            continue
    return ""


def _best_response_candidate(candidates: list[str]) -> str:
    best = ""
    for text in candidates:
        cleaned = text.strip()
        if not cleaned or is_noise_response(cleaned):
            continue
        if len(cleaned) > len(best):
            best = cleaned
    return best


def extract_latest_response(page: Page) -> str:
    candidates: list[str] = []

    for selector in ASSISTANT_MESSAGE_SELECTORS:
        messages = page.locator(selector)
        count = messages.count()
        for i in range(count - 1, -1, -1):
            text = _element_text(messages.nth(i))
            if text:
                candidates.append(text)

    best = _best_response_candidate(candidates)
    if best:
        return best

    try:
        js_text = page.evaluate(
            """() => {
                const selectors = [
                    '.model-response-text',
                    'message-content',
                    '[data-message-author-role="model"]',
                    '.response-container',
                    'model-response',
                ];
                const nodes = [];
                for (const selector of selectors) {
                    document.querySelectorAll(selector).forEach((node) => nodes.push(node));
                }
                if (!nodes.length) return '';
                const last = nodes[nodes.length - 1];
                return (last.textContent || last.innerText || '').trim();
            }"""
        )
        if isinstance(js_text, str) and js_text.strip():
            candidates.append(js_text.strip())
    except Exception:
        pass

    best = _best_response_candidate(candidates)
    if best:
        return best

    for text in reversed(page.locator("main").locator("p, div, span").all_inner_texts()):
        cleaned = text.strip()
        if cleaned and not is_noise_response(cleaned):
            candidates.append(cleaned)

    return _best_response_candidate(candidates)


def wait_for_response(page: Page, timeout_ms: int) -> str:
    print("Waiting for Gemini response...")
    page.wait_for_timeout(1500)

    deadline = time.time() + (timeout_ms / 1000)
    last_text = ""
    stable_count = 0

    while time.time() < deadline:
        generating = is_generating(page)
        current_text = extract_latest_response(page)
        json_incomplete = (
            looks_like_json_response(current_text)
            and not json_response_complete(current_text)
        )
        if (
            current_text
            and current_text == last_text
            and not generating
            and not json_incomplete
        ):
            stable_count += 1
            if stable_count >= 3:
                return current_text
        else:
            stable_count = 0
            last_text = current_text

        page.wait_for_timeout(1000)

    if last_text and not is_noise_response(last_text):
        print("Warning: timed out waiting for response to stabilize; returning partial text.")
        return last_text

    raise RuntimeError("No response received within the timeout period.")


def save_screenshot(page: Page, output_dir: Path, filename: str = "last_response.png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / filename
    page.screenshot(path=str(screenshot_path), full_page=True)
    return screenshot_path


def run_stage(
    page: Page,
    *,
    model_key: str,
    prompt: str,
    image_path: Path | None = None,
    timeout_ms: int = 120_000,
) -> str:
    """Run one isolated pipeline stage in a fresh chat."""
    start_new_chat(page)
    select_model(page, model_key)
    if image_path is not None:
        upload_image(page, image_path)
    send_prompt(page, prompt)
    return wait_for_response(page, timeout_ms)
