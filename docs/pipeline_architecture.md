# ATSN — Accessible Alt-Text Generation Pipeline

> **Runtime note:** This public repository runs all pipeline and evaluator stages via the **OpenAI API** ([`backend/openai.py`](../src/atsn/backend/openai.py)). Gemini model names referenced below are from the original research design; actual model IDs are configured in [`backend/config.py`](../src/atsn/backend/config.py).

**Purpose:** Generate high-quality, accurate, screen-reader-friendly **ALT text** for e-commerce product images, serving Blind and Low-Vision (BLV) users. The system takes a product image plus its surrounding page text and produces a final, objective, hallucination-free ALT text through a multi-stage generate → deconstruct → validate → fuse pipeline.

This document is the single source of truth for the pipeline: its architecture, the role of every stage, the standardized data contract that connects them, and the recommended model tier for each stage.

---

## 1. High-Level Architecture

The pipeline has **7 prompt-driven stages**. Stages 4–6 (the three validators) run **in parallel**; every other stage is sequential.

```
                                  ┌──────────────────────────────────────────────┐
                                  │            PARALLEL VALIDATION                 │
                                  │                                                │
 [1] Classifier ─► [2] Generator ─► [3] ALT-to-List ─┬─► [4] Accuracy Validator ──┐│
   (routes to        (clothing or     (atomic claims) │                            ││
    clothing or       furniture)                      ├─► [5] Completeness Val. ───┼┼─► [7] Fuser ─► final_alt_text
    furniture)                                        │                            ││
                                                      └─► [6] Redundancy Validator ┘│
                                  └──────────────────────────────────────────────┘
```

- **Input to the pipeline:** a product image + `SURROUNDING_TEXT` (title, description, specs). A normalized JSON form of that text, `PRODUCT_DOM`, is used by the validators.
- **Output of the pipeline:** a single polished `final_alt_text` string (plus a full audit trail of claim-level decisions).
- **Category scope:** every product is either **CLOTHING** or **FURNITURE**. There are no other categories.

---

## 2. The Standardized Data Contract

All JSON-producing stages emit a top-level `"stage"` field identifying themselves. All template inputs use the `{{DOUBLE_BRACE}}` convention with the canonical token names below.

### Canonical input tokens

| Token | Meaning | Produced by | Consumed by |
|---|---|---|---|
| `{{SURROUNDING_TEXT}}` | Raw product page text (title, description, specs) | Pipeline input | Classifier, Generator |
| `{{PRODUCT_DOM}}` | Normalized JSON of the product info | Pipeline input | Redundancy |
| `{{ALT_TEXT}}` | The generated ALT text paragraph | Generator (stage 2) | ALT-to-List, Fuser |
| `{{ATOMIC_CLAIMS}}` | JSON array of atomic claims `[{ "claim_id", "claim" }]` | ALT-to-List (stage 3) | Accuracy, Completeness, Redundancy, Fuser |
| `{{ACCURACY_OUTPUT}}` | Accuracy validator JSON | Accuracy (stage 4) | Fuser |
| `{{COMPLETENESS_OUTPUT}}` | Completeness validator JSON | Completeness (stage 5) | Fuser |
| `{{REDUNDANCY_OUTPUT}}` | Redundancy validator JSON | Redundancy (stage 6) | Fuser |

### The shared claim object

The atom that flows through the whole pipeline:

```json
{ "claim_id": "c1", "claim": "<one independently checkable visual fact>" }
```

- Original claims are numbered `c1, c2, c3 …` (reading order), assigned by the ALT-to-List stage.
- The Completeness validator appends **new** claims numbered `n1, n2, …`.
- Every stage **joins strictly by `claim_id`** — never by matching claim text.

### `"stage"` identifiers

| Stage | `"stage"` value |
|---|---|
| Classifier | `classification` |
| Generator | *(plain text output — no JSON)* |
| ALT-to-List | `claim_extraction` |
| Accuracy Validator | `accuracy_validation` |
| Completeness Validator | `completeness_validation` |
| Redundancy Validator | `redundancy_validation` |
| Fuser | `fusion` |

---

## 3. The Stages (A–Z)

### Stage 1 — Classifier
- **File:** `prompts/classifier.txt`
- **Job:** Route the product to exactly one category: **CLOTHING** or **FURNITURE**.
- **Inputs:** `{{SURROUNDING_TEXT}}` + product image.
- **Logic:** Text-first (the stated product type is the primary signal); the image confirms, disambiguates, or acts as fallback when text is vague/missing. Emits a `confidence`, `text_image_agreement`, and `needs_review` flag.
- **Output:** JSON with `category`, `confidence`, `primary_signal`, `text_image_agreement`, `alternative_category`, `needs_review`, `evidence`.
- **Recommended model:** **`gemini-3.5-flash-lite`** — a binary text-first decision; cheapest tier that still reads the image reliably.

### Stage 2 — Generator (Clothing / Furniture)
- **Files:** `prompts/clothing_generator.txt`, `prompts/furniture_generator.txt`
- **Job:** Produce the initial ALT text. The Classifier's output selects which of the two generators runs.
- **Inputs:** `{{SURROUNDING_TEXT}}` + product image.
- **Key rules:**
  - **Length ceiling:** 2–4 sentences (~30–55 words). Shorter is fine; never pad.
  - **Anti-duplication:** don't repeat what the user already heard in the surrounding text — only the 3 exceptions (Subject Anchoring, Spatial Mapping, Visual Translation).
  - **Objectivity mandate:** no evaluative/aesthetic/subjective words.
  - **Recognizable subjects & on-product text:** name confidently identifiable entities; translate/convey the meaning of non-English text.
  - **Clothing-specific:** (a) *Garment identity* — name a confidently recognizable type/team/brand kit (e.g., "FC Barcelona home football kit," "basketball uniform," "two-piece pajama set") from visual evidence; (b) *Multi-color rule* — the DOM usually names only the main color, so read all prominent colors and their placement from the pixels (color-blocks, contrast sleeves/collar/trim).
  - **Furniture-specific:** *Secondary/nearby objects* — mention staging props on or near the product ONLY to convey scale, capacity/use, or the product's own layout; refer to them briefly/collectively without their own detail, and stay focused on the main product.
  - **Hallucination alerts** + "omit, don't announce" + silent cropping boundary.
- **Output:** plain ALT text prose (no labels/headers).
- **Recommended model:** **`gemini-3.1-pro-preview`** — quality-defining stage; demands the strongest vision + complex rubric adherence. Everything downstream inherits this output.

### Stage 3 — ALT-to-List (Claim Deconstructor)
- **File:** `prompts/alt_to_list.txt`
- **Job:** Split `{{ALT_TEXT}}` into atomic visual claims — one independently checkable fact each — preserving reading order.
- **Inputs:** `{{ALT_TEXT}}` (text only, no image).
- **Key rules:** feature-level atomicity (bundle an entity's own color/shape/text/position together; split distinct features apart); faithful extraction (add nothing, delete nothing, don't normalize values); ignore framing/meta phrases.
- **Output:** `{ "stage": "claim_extraction", "claims": [...], "total_claims": N }`. The `claims` array is the canonical `ATOMIC_CLAIMS` payload.
- **Recommended model:** **`gemini-3.5-flash-lite`** — text-only, mechanical, rule-based decomposition.

### Stage 4 — Accuracy Validator (False-Positive check)
- **File:** `prompts/accuracy_validator.txt`
- **Job:** The pipeline's **primary hallucination filter**. Audit each claim against the **image**: DELETE features that are not actually visible (invented, text-seeded, or occluded), CORRECT features that are visible but described wrongly, and DELETE unprovable subjective fluff. The image is the sole ground truth — never the claim's wording or what the product "usually" has. Enforces an **existence gate** that defaults to DELETE under uncertainty.
- **Inputs:** product image + `{{ATOMIC_CLAIMS}}`.
- **Labels:** `ACCURATE` | `CORRECTED` | `DELETED`. Error categories: `NONEXISTENT`, `ATTRIBUTE_OR_OCR`, `SPATIAL`, `STATE`, `ARTIFACT`, `SUBJECTIVE`.
- **Output:** per-claim verdicts (`original_claim`, `claim`, `label`, `error_category`, `reason`) + `summary` with `error_rate`.
- **Recommended model:** **`gemini-3.1-pro-preview`** — the most pixel-demanding verification; errors let hallucinations survive or delete true claims.

### Stage 5 — Completeness Validator (False-Negative check)
- **File:** `prompts/completeness_validator.txt`
- **Job:** Find prominent visual features that are **missing** from the claim list and append them. Keeps existing claims untouched. Judges purely from the image — no product text.
- **Inputs:** product image + `{{ATOMIC_CLAIMS}}`.
- **Labels:** `RETAINED` (existing, unchanged) | `ADDED` (new, id `n1, n2, …`). Addition types: `VISUAL_GAP`, `NON_ENGLISH_TEXT`, `KNOWN_ENTITY`.
- **Output:** full list (retained first, then added) + `summary` with `omission_score`.
- **Recommended model:** **`gemini-3.1-pro-preview`** — spotting what's missing is vision-heavy. *(Budget option: `gemini-3.7-flash`.)*

### Stage 6 — Redundancy Validator
- **File:** `prompts/redundancy_validator.txt`
- **Job:** Decide whether each claim adds information beyond the `PRODUCT_DOM` or **needlessly repeats** what the user already has. **Semantic** redundancy only — not lexical.
- **Inputs:** product image + `{{PRODUCT_DOM}}` + `{{ATOMIC_CLAIMS}}`. (Image used only to test whether repeated DOM info performs a real visual-grounding function.)
- **Labels:** `NOVEL` | `UNAVOIDABLE_REPETITION` | `AVOIDABLE_REDUNDANCY`.
- **Output:** per-claim label + `reason` + `summary` with `redundancy_score` (= avoidable / total).
- **Recommended model:** **`gemini-3.7-flash`** — mostly semantic text reasoning; image only lightly used.

### Stage 7 — Fuser
- **File:** `prompts/fuser.txt`
- **Job:** Deterministically compile the three validator verdicts into one clean, accurate ALT text. Acts as a **compiler, not an author** — never invents content; only grammar/word-order.
- **Inputs:** `{{ALT_TEXT}}` + `{{ATOMIC_CLAIMS}}` + `{{ACCURACY_OUTPUT}}` + `{{COMPLETENESS_OUTPUT}}` + `{{REDUNDANCY_OUTPUT}}`. **No image.**
- **Resolution hierarchy (strict order):**
  1. **Kill-switch:** Accuracy `DELETED` → drop permanently (`DROPPED_HALLUCINATION`).
  2. **Content authority:** Accuracy's `claim` text is authoritative (corrected if `CORRECTED`).
  3. **Redundancy:** `AVOIDABLE_REDUNDANCY` → drop (`DROPPED_REDUNDANT`), *unless* Accuracy `CORRECTED` it (then keep). `NOVEL`/`UNAVOIDABLE_REPETITION` → keep.
  4. **Additions:** integrate Completeness `ADDED` claims (`MERGED` if duplicating a survivor).
  - **Tie-break:** missing verdict → trust Accuracy; if that's missing too → retain original claim text.
- **Output:** `{ "stage": "fusion", "master_claims": [...], "edit_log": [...], "final_alt_text": "..." }`. Includes every evaluated `claim_id` (surviving and dropped) for auditability.
- **Recommended model:** **`gemini-3.7-flash`** — text-only, deterministic multi-step compile + grammar healing; no vision needed.

---

## 4. Model Assignment Summary

| # | Stage | File | Model (Model ID) | Uses image? |
|---|---|---|---|---|
| 1 | Classifier | `classifier.txt` | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | Yes (secondary) |
| 2 | Generator | `clothing_generator.txt` / `furniture_generator.txt` | Gemini 3.1 Pro (`gemini-3.1-pro-preview`) | Yes (primary) |
| 3 | ALT-to-List | `alt_to_list.txt` | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | No |
| 4 | Accuracy Validator | `accuracy_validator.txt` | Gemini 3.1 Pro (`gemini-3.1-pro-preview`) | Yes (primary) |
| 5 | Completeness Validator | `completeness_validator.txt` | Gemini 3.1 Pro (`gemini-3.1-pro-preview`) | Yes (primary) |
| 6 | Redundancy Validator | `redundancy_validator.txt` | Gemini 3.7 Flash (`gemini-3.7-flash`) | Yes (light) |
| 7 | Fuser | `fuser.txt` | Gemini 3.7 Flash (`gemini-3.7-flash`) | No |

**Rationale in one line:** reserve **Pro** for the vision-critical, quality-defining stages (generation + the two visual validators); use **Flash** for text-reasoning stages (redundancy, fusion); use **Flash-Lite** for the trivial ones (classification, extraction).

### Why 3.7 Flash rather than 3.8 Flash
For these **single-shot** prompts, 3.8 Flash's improvements are concentrated in long-horizon agentic workflows, and it burns roughly **2× the thinking tokens** for essentially flat quality on non-agentic tasks. 3.7 Flash delivers the same result at lower real cost. Upgrade to `gemini-3.8-flash` only if you observe reasoning misses.

---

## 5. Recommended Runtime Settings

| Stage | Temperature | Thinking effort | Notes |
|---|---|---|---|
| Classifier | 0 | low | Deterministic routing. |
| Generator | 0.2–0.4 | medium/high | Slight latitude for natural phrasing; strong vision reasoning. |
| ALT-to-List | 0 | low | Deterministic, mechanical split. |
| Accuracy Validator | 0 | medium/high | Careful pixel-level verification. |
| Completeness Validator | 0 | medium/high | Thorough gap-finding. |
| Redundancy Validator | 0 | medium | Semantic reasoning. |
| Fuser | 0 | low | Prompt **requires** determinism (identical inputs → identical output). |

- All Gemini 3 models are **multimodal** with a **1M-token context window**, so image input works on every tier.
- All stages demand **strict JSON only** (except the Generator, which returns plain prose). Enforce with response schema / JSON mode where available.

---

## 6. Cost Notes (as of Sept 2026)

Per 1M tokens (input / output):

| Model | Input | Output |
|---|---|---|
| Gemini 3.1 Pro (`gemini-3.1-pro-preview`) | $2.00 | $12.00 (tiered higher >200K-token prompts) |
| Gemini 3.7 Flash (`gemini-3.7-flash`) | $0.75 | $3.75 |
| Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | $0.30 | $2.50 |

> The $0.75 / $3.75 Flash rate is **introductory through Dec 31, 2026**; it rises to **$1.50 / $7.50 on Jan 1, 2027**. Factor this into volume budgeting. Batch, cached, and grounding rates differ — check Google's live pricing page for production.

### Whole-pipeline dials
- **Budget mode:** run Generator + Accuracy + Completeness on `gemini-3.7-flash`; keep the rest on Flash-Lite. Cheapest viable pipeline; expect some loss in fine visual accuracy.
- **Max-quality mode:** run Generator + all three validators on `gemini-3.1-pro-preview`; keep Extractor/Fuser/Classifier on Flash/Flash-Lite (they gain nothing from Pro's vision).

---

## 7. File Map

All prompt files live under `prompts/`.

| File | Stage | Output type |
|---|---|---|
| `classifier.txt` | 1 — Classifier | JSON |
| `clothing_generator.txt` | 2 — Generator (clothing) | Plain ALT text |
| `furniture_generator.txt` | 2 — Generator (furniture) | Plain ALT text |
| `alt_to_list.txt` | 3 — Claim Deconstructor | JSON |
| `accuracy_validator.txt` | 4 — Accuracy Validator | JSON |
| `completeness_validator.txt` | 5 — Completeness Validator | JSON |
| `redundancy_validator.txt` | 6 — Redundancy Validator | JSON |
| `fuser.txt` | 7 — Fuser | JSON (contains `final_alt_text`) |
| `evaluators/relevancy.txt` | Post-pipeline evaluation | JSON |
| `evaluators/redundancy.txt` | Post-pipeline evaluation | JSON |
| `evaluators/objectivity.txt` | Post-pipeline evaluation | JSON |
| `docs/pipeline_architecture.md` | — | This document |

---

## 8. End-to-End Example Flow

1. **Classifier** reads title *"New Era NFL Hoody…"* + image → `{ "category": "CLOTHING", ... }`.
2. **Clothing Generator** produces ALT text, e.g. *"Front view of a heather-gray hoodie with the NFL shield logo centered on the chest…"*.
3. **ALT-to-List** → `[{c1: "heather-gray hoodie"}, {c2: "NFL shield logo centered on the chest"}, …]`.
4. **Accuracy / Completeness / Redundancy** each evaluate that claim list in parallel against the image and DOM.
5. **Fuser** merges the three verdicts by `claim_id` using the resolution hierarchy → clean `final_alt_text` + `edit_log` audit trail.
