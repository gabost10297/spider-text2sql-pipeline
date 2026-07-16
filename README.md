# spider-text2sql-pipeline

Autonomous Text-to-SQL LLM agent pipeline on the [Spider](https://yale-lily.github.io/spider) benchmark: schema linking → QLoRA fine-tuning → execution accuracy → agentic self-correction → FastAPI + Streamlit.

Phases 1–2 run entirely in **Docker** (no local venv required).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Compose v2
- ~5–10 GB free disk for image + Spider archives + Hugging Face cache

## Phase 1 — Environment & EDA

### 1. Build the image

```powershell
docker compose build phase1
```

### 2. Download Spider

Fetches question/SQL pairs from Hugging Face (`xlangai/spider`) and SQLite databases from the `HAL-9001/spider-databases` mirror (Yale Spider, CC-BY-SA-4.0). Assets land under `./data` on the host.

```powershell
docker compose run --rm phase1 python -m src.phase1.download_spider
```

Re-download / overwrite:

```powershell
docker compose run --rm phase1 python -m src.phase1.download_spider --force
```

### 3. Explore the dataset (CLI)

```powershell
docker compose run --rm phase1 python -m src.phase1.explore_spider
```

Prints split sizes, a sample record (`question` / `db_id` / `query`), complexity signals, and a peek at `concert_singer.sqlite` when databases are present.

### 4. Browse the data (UI)

Streamlit app with five pages: Splits Browser, Schema Explorer, SQLite Query Runner, Linked Schemas, and Prompt Viewer.

```powershell
docker compose up -d browser
```

Open [http://localhost:8501](http://localhost:8501).

Stop when done:

```powershell
docker compose down
```

Requires the download step first; without data the home page shows a warning.

### Interactive shell

```powershell
docker compose --profile tools run --rm shell
```

## Phase 2 — Schema linking & prompt formulation

Requires Phase 1 data under `./data` (parquet splits + SQLite databases).

### 1. Schema linker

Serialises each SQLite catalog (tables, columns, types, foreign keys) into prompt text.

```powershell
# one database → stdout
docker compose run --rm phase1 python -m src.phase2.schema_linker concert_singer

# one database → file
docker compose run --rm phase1 python -m src.phase2.schema_linker concert_singer --out data/spider/schemas/concert_singer.txt

# every database → one {db_id}.txt each
docker compose run --rm phase1 python -m src.phase2.schema_linker --all --out data/spider/schemas
```

### 2. Instruction prompt templates

Builds Llama-3 Instruct chat prompts (system + schema/question + `SQL: …`).

```powershell
# preview train[0] (training prompt with golden SQL)
docker compose run --rm phase1 python -m src.phase2.prompts --from-parquet train --index 0

# inference-style (no golden SQL in the prompt)
docker compose run --rm phase1 python -m src.phase2.prompts --from-parquet train --index 0 --inference

# ad-hoc
docker compose run --rm phase1 python -m src.phase2.prompts --db-id concert_singer --question "How many singers do we have?"
```

Python API (inside the container / `python` REPL, not PowerShell):

```python
from src.phase2.prompts import generate_training_prompt, generate_prompt_for_example
```

### 3. Training records

Export formatted examples for train + validation (JSONL: `db_id`, `question`, `query`, `prompt`):

```powershell
docker compose run --rm phase1 python -m src.phase2.prompts --export-all
```

Single split:

```powershell
docker compose run --rm phase1 python -m src.phase2.prompts --export-split train
docker compose run --rm phase1 python -m src.phase2.prompts --export-split validation
```

## Data layout

```text
data/
  spider/
    hf_export/              # train/validation parquet
    tables.json
    spider_data.zip         # cached archive
    .download_complete
    schemas/                # Phase 2: {db_id}.txt schema dumps
    prompts/                # Phase 2: train.jsonl, validation.jsonl
  databases/
    {db_id}/{db_id}.sqlite  # same path convention as the FastAPI blueprint
```

## Project layout

```text
Dockerfile
docker-compose.yml
requirements.txt            # Phase 1–2 deps (incl. Streamlit)
src/phase1/
  download_spider.py
  explore_spider.py
  paths.py
  browser/
    Home.py
    pages/
      1_Splits_Browser.py
      2_Schema_Explorer.py
      3_SQLite_Query.py
      4_Linked_Schemas.py
      5_Prompt_Viewer.py
src/phase2/
  schema_linker.py
  prompts.py
scripts/phase1.sh
```

## Next (Phase 3)

Parameter-efficient fine-tuning (QLoRA / Unsloth) on the exported prompts, including a PyTorch dataset with loss masking (`-100` on prompt tokens, loss only on SQL). Needs a GPU compose profile.
