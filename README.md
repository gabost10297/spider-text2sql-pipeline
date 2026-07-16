# spider-text2sql-pipeline

Autonomous Text-to-SQL LLM agent pipeline on the [Spider](https://yale-lily.github.io/spider) benchmark: schema linking → QLoRA fine-tuning → execution accuracy → agentic self-correction → FastAPI + Streamlit.

Phase 1 runs entirely in **Docker** (no local venv required).

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

Streamlit app with three pages: Splits Browser, Schema Explorer, and SQLite Query Runner.

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

## Data layout

```text
data/
  spider/
    hf_export/              # train/validation parquet
    tables.json
    spider_data.zip         # cached archive
    .download_complete
  databases/
    {db_id}/{db_id}.sqlite  # same path convention as the FastAPI blueprint
```

## Project layout

```text
Dockerfile
docker-compose.yml
requirements.txt            # Phase 1 deps (incl. Streamlit)
src/phase1/
  download_spider.py
  explore_spider.py
  paths.py
  browser/
    Home.py                 # Streamlit entry point
    pages/
      1_Splits_Browser.py
      2_Schema_Explorer.py
      3_SQLite_Query.py
scripts/phase1.sh
```

## Next (Phase 2)

Automated schema linking + Llama-3 instruction prompt templates, still runnable via Docker (GPU compose profile comes with Phase 3 / QLoRA).
