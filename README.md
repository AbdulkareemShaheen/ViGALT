# ViGALT — Visual-Grounded ALT Text Generation

Research codebase for **ViGALT**, a multi-stage pipeline that generates high-quality, screen-reader-friendly alt text for e-commerce product images. The system takes a product image plus surrounding page text and produces objective, hallucination-filtered alt text through a 7-stage generate → deconstruct → validate → fuse pipeline.

This repository is the public release accompanying our research paper. It includes the full source code, prompts, product metadata, and the consolidated evaluation dataset with published results for **ViGALT** vs. the **ASSETS24** baseline.

## Pipeline Overview

```mermaid
flowchart LR
  classifier[Classifier] --> generator[Generator]
  generator --> altToList[ALT-to-List]
  altToList --> accuracy[Accuracy Validator]
  altToList --> completeness[Completeness Validator]
  altToList --> redundancy[Redundancy Validator]
  accuracy --> fuser[Fuser]
  completeness --> fuser
  redundancy --> fuser
  fuser --> finalAlt[final_alt_text]
```

Every product is classified as **CLOTHING** or **FURNITURE**. Stages 4–6 run in parallel; all other stages are sequential. See [docs/pipeline_architecture.md](docs/pipeline_architecture.md) for the full specification.

## Repository Layout

```
.
├── src/atsn/              Python package (pipeline + evaluators)
├── prompts/               Stage prompts and evaluation rubrics
├── data/products/         30 product JSON files (15 clothing, 15 furniture)
├── evaluation_dataset/    Published per-image results (30 folders + summary.json)
├── docs/                  Architecture documentation
├── pyproject.toml         Package metadata
├── requirements.txt       Pinned dependencies
└── .env.example           API key template
```

### `src/` structure

| Group | Modules |
|---|---|
| Pipeline core | `pipeline.py`, `pipeline_utils.py`, `pipeline_types.py` |
| Backend | `openai_backend.py`, `openai_config.py`, `openai_schemas.py` |
| DOM extraction | `extract_dom.py`, `amazon_extractor.py` |
| Orchestration | `run_from_url.py` |
| Evaluators | `*_evaluator_batch.py`, `combine_claims_relevancy.py`, `evaluator_batch_utils.py` |
| Dataset tools | `build_dataset.py`, `alt_to_list_batch.py` |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and set your API key:

```
OPENAI_API_KEY=sk-...
```

---

## Quick start: one Amazon URL → full run

Runs DOM extraction, alt-text generation, claim extraction, and all four evaluation metrics:

```bash
python -m atsn.run_from_url --url "https://www.amazon.fr/dp/B077XM3DV5"
```

Outputs go to `output/runs/B077XM3DV5/` (DOM JSON, image, evaluation JSONs). Pipeline output is saved to `output/B077XM3DV5_pipeline.json`.

Options:

```bash
python -m atsn.run_from_url --url "..." --work-dir output/runs/my_product
python -m atsn.run_from_url --html saved_page.html --work-dir output/runs/B077XM3DV5
python -m atsn.run_from_url --url "..." --skip-eval          # pipeline only
python -m atsn.run_from_url --url "..." --single-model gpt-4o
```

---

## Step-by-step guide

### Step 1 — Extract product DOM + image

Build a product JSON from an Amazon URL. Downloads the main product image into the same folder.

```bash
python -m atsn.extract_dom \
  --url "https://www.amazon.fr/dp/B077XM3DV5" \
  --output-dir output/runs/B077XM3DV5
```

Creates:
- `output/runs/B077XM3DV5/B077XM3DV5.json` — title, brand, description, feature bullets, product details, `main_image`
- `output/runs/B077XM3DV5/B077XM3DV5.jpg` — downloaded product image

If Amazon blocks automated requests, save the page HTML in your browser and parse locally:

```bash
python -m atsn.extract_dom --html saved_page.html --output-dir output/runs/B077XM3DV5
```

### Step 2 — Generate alt text (7 pipeline stages)

One command runs all generation stages via OpenAI:

| Order | Stage | Prompt | Input |
|---|---|---|---|
| 1 | Classifier | `prompts/classifier.txt` | image + surrounding text |
| 2 | Generator | `prompts/clothing_generator.txt` or `furniture_generator.txt` | image + surrounding text |
| 3 | Claim extraction | `prompts/alt_to_list.txt` | generated alt text |
| 4 | Accuracy validator | `prompts/accuracy_validator.txt` | image + claims |
| 5 | Completeness validator | `prompts/completeness_validator.txt` | image + claims |
| 6 | Redundancy validator | `prompts/redundancy_validator.txt` | image + claims |
| 7 | Fuser | `prompts/fuser.txt` | validator outputs |

```bash
python -m atsn.pipeline --product output/runs/B077XM3DV5/B077XM3DV5.json
```

Output: `output/B077XM3DV5_pipeline.json` with `final_alt_text` and per-stage results.

### Step 3 — Extract claims (evaluation format)

The pipeline already runs claim extraction inline (stage 3). This step re-extracts claims into the batch format used by evaluators:

```bash
python -m atsn.alt_to_list_batch \
  --source pipeline \
  --pipeline output/B077XM3DV5_pipeline.json \
  --output-dir output/runs/B077XM3DV5/final_alt_claim_lists
```

Output: `output/runs/B077XM3DV5/final_alt_claim_lists/B077XM3DV5_claims.json`

### Step 4 — Relevancy evaluation

```bash
python -m atsn.relevancy_evaluator_batch \
  --algorithm our \
  --our-claims-dir output/runs/B077XM3DV5/final_alt_claim_lists \
  --products-dir output/runs/B077XM3DV5 \
  --images-dir output/runs/B077XM3DV5 \
  --output output/runs/B077XM3DV5/relevancy_evaluations.json
```

Prompt: `prompts/evaluators/relevancy.txt`

### Step 5 — Redundancy evaluation

```bash
python -m atsn.redundancy_evaluator_batch \
  --algorithm our \
  --relevancy-input output/runs/B077XM3DV5/relevancy_evaluations.json \
  --products-dir output/runs/B077XM3DV5 \
  --output output/runs/B077XM3DV5/redundancy_evaluations.json
```

Prompt: `prompts/evaluators/redundancy.txt`

### Step 6 — Objectivity evaluation

```bash
python -m atsn.objectivity_evaluator_batch \
  --algorithm our \
  --relevancy-input output/runs/B077XM3DV5/relevancy_evaluations.json \
  --products-dir output/runs/B077XM3DV5 \
  --images-dir output/runs/B077XM3DV5 \
  --output output/runs/B077XM3DV5/objectivity_evaluations.json
```

Prompt: `prompts/evaluators/objectivity.txt`

### Step 7 — Efficiency evaluation

Deterministic metric (no LLM): `(relevant_novel_claims / ALT word count) × 100`

```bash
python -m atsn.efficiency_evaluator_batch \
  --algorithm our \
  --redundancy-input output/runs/B077XM3DV5/redundancy_evaluations.json \
  --relevancy-input output/runs/B077XM3DV5/relevancy_evaluations.json \
  --output output/runs/B077XM3DV5/efficiency_evaluations.json
```

---

## Batch reproduction (30 products)

### What is already included

The `evaluation_dataset/` folder contains everything needed to **verify** the paper numbers:

| File per folder (`1/` … `30/`) | Contents |
|---|---|
| `<image>.jpg` | Product image |
| `dom.json` | Product metadata (title, brand, features, …) |
| `alt_text_and_claims.txt` | ViGALT and ASSETS24 alt text + plain claims |
| `evaluation.json` | Relevancy, redundancy, objectivity, efficiency metrics |

Top-level `evaluation_dataset/summary.json` holds averaged metrics across all 30 products.

### Re-run pipeline on all products

```bash
python -m atsn.pipeline --all
python -m atsn.pipeline --single-model gpt-4o --product data/products/1_clothing.json
```

### Re-run evaluators on all products

```bash
python -m atsn.alt_to_list_batch --source pipeline --all
python -m atsn.relevancy_evaluator_batch
python -m atsn.redundancy_evaluator_batch
python -m atsn.objectivity_evaluator_batch
python -m atsn.efficiency_evaluator_batch
python -m atsn.combine_claims_relevancy
```

### Rebuild evaluation dataset from raw `Dataset/` folders

```bash
python -m atsn.build_dataset --source Dataset --output evaluation_dataset
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Relevancy | Whether each atomic claim is relevant (R), supplementary (S), or visual-only (V) |
| Redundancy | Novel vs. avoidable repetition relative to product DOM |
| Objectivity | Objective vs. subjective claim labels |
| Efficiency | `(relevant_novel_claims / ALT word count) × 100` |

## Dataset

- **30 products:** 15 clothing (`1_clothing` … `15_clothing`) and 15 furniture (`16_furniture` … `30_furniture`)
- Each product JSON in `data/products/` contains title, brand, description, feature bullets, and a `main_image` URL
- Published evaluation results are in `evaluation_dataset/` with `summary.json` averages

## Notes

- Amazon DOM extraction may be blocked by bot detection; use `--html` with a saved page when needed. Scraping is your responsibility under Amazon's terms of service.
- Re-running the pipeline produces new alt texts; LLM outputs vary. Use `evaluation_dataset/` to verify published paper numbers.
- If you see `atsn.egg-info/` under `src/` after `pip install -e .`, that is a local build artifact (gitignored).

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{vigalt2026,
  title   = {ViGALT: Visual-Grounded ALT Text Generation for E-Commerce Product Images},
  author  = {TODO: Add authors},
  journal = {TODO: Add venue},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
