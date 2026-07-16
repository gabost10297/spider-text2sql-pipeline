"""PyTorch Dataset for Text-to-SQL with prompt loss masking.

WHY MASK LABELS?
----------------
Causal LMs predict the *next* token. If we train on the full string
(system + schema + question + SQL), the model also spends capacity
memorising instructions and schemas. We only want gradients on the SQL.

CrossEntropyLoss ignores target id ``-100`` (PyTorch convention). So:

    labels[i] = -100          → no loss (prompt / padding)
    labels[i] = input_ids[i]  → learn to predict this token (SQL + EOT)

HOW WE FIND THE CUT POINT
-------------------------
Phase 2 can build two strings:

1. *prompt_prefix* — ends at ``SQL: `` (inference-style, no golden SQL)
2. *full_text*     — same prefix + ``{sql}<|eot_id|>``

We tokenize both with ``add_special_tokens=False``. Ideally
``full_ids`` starts with ``prefix_ids``. Then:

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]

If BPE makes the prefix *not* an exact ID prefix of the full sequence
(rare but possible), we fall back to searching for the token pattern of
``SQL: `` inside ``full_ids`` and mask through that marker.

THIS MODULE
-----------
- ``build_masked_example`` — one string → tensors
- ``Text2SqlDataset`` — Spider parquet / JSONL → examples
- CLI smoke test — print mask stats + decode supervised span
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset

from src.phase1.paths import HF_EXPORT_DIR, SPIDER_ROOT
from src.phase2.prompts import generate_training_prompt, get_schema
from src.phase3.tokenizer import encode_text, load_tokenizer

console = Console()

IGNORE_INDEX = -100
PROMPTS_DIR = SPIDER_ROOT / "prompts"


def prompt_cut_index(
    tokenizer,
    prefix_text: str,
    full_ids: list[int],
) -> tuple[int, str]:
    """Index where supervised (SQL) tokens begin in *full_ids*.

    Returns (cut_index, method) where method describes how the cut was found.
    Tokens ``full_ids[:cut]`` should be masked with -100.

    Note: tokenizing ``\"SQL: \"`` alone often ends with a lone space token
    (``Ġ``), while the full string merges that space into ``ĠSELECT``.
    We therefore prefer matching a rstrip'd prefix, then fall back to
    locating the last ``SQL`` + ``:`` token pair in the full sequence.
    """
    # 1) Prefix without trailing whitespace — usually aligns through ``SQL:``
    prefix_ids = encode_text(tokenizer, prefix_text.rstrip())["token_ids"]
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return len(prefix_ids), "prefix_rstrip"

    # 2) Exact prefix (rarely works when trailing space merges into next word)
    prefix_ids_raw = encode_text(tokenizer, prefix_text)["token_ids"]
    if full_ids[: len(prefix_ids_raw)] == prefix_ids_raw:
        return len(prefix_ids_raw), "prefix"

    # 3) Last ``SQL`` + ``:`` pair (assistant turn marker in our templates)
    sql_id = tokenizer.convert_tokens_to_ids("SQL")
    colon_id = tokenizer.convert_tokens_to_ids(":")
    last = -1
    for i in range(len(full_ids) - 1):
        if full_ids[i] == sql_id and full_ids[i + 1] == colon_id:
            last = i
    if last >= 0:
        return last + 2, "sql_colon"

    raise ValueError(
        "Could not align prompt prefix with full token ids. "
        "Check that prefix_text is exactly the start of the training string."
    )


def build_masked_example(
    tokenizer,
    *,
    schema: str,
    question: str,
    sql: str,
    max_length: int | None = 2048,
) -> dict[str, Any]:
    """Tokenize one training example and apply -100 loss masking.

    Returns dict with ``input_ids``, ``attention_mask``, ``labels`` (lists),
    plus debug fields ``cut``, ``method``, ``n_prompt``, ``n_supervised``.
    """
    prefix_text = generate_training_prompt(schema, question, sql=None)
    full_text = generate_training_prompt(schema, question, sql=sql)

    full_ids = encode_text(tokenizer, full_text)["token_ids"]
    cut, method = prompt_cut_index(tokenizer, prefix_text, full_ids)

    if max_length is not None and len(full_ids) > max_length:
        # Truncate from the left? For SQL we prefer keeping the end (SQL).
        # Simpler teaching default: truncate the right (drop tail) and warn.
        full_ids = full_ids[:max_length]
        cut = min(cut, len(full_ids))

    labels = [IGNORE_INDEX] * cut + full_ids[cut:]
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "cut": cut,
        "method": method,
        "n_prompt": cut,
        "n_supervised": len(full_ids) - cut,
        "n_tokens": len(full_ids),
        "full_text": full_text,
    }


class Text2SqlDataset(Dataset):
    """Map-style dataset over Spider train/validation examples."""

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "train",
        source: str = "parquet",
        max_length: int | None = 2048,
        limit: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split:
            ``train`` or ``validation``.
        source:
            ``parquet`` — build prompts live via schema linker.
            ``jsonl`` — use exported ``data/spider/prompts/{split}.jsonl``
            (still re-builds prefix/full for correct masking).
        max_length:
            Truncate sequences longer than this (tokens).
        limit:
            Optional cap on number of rows (fast smoke tests).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows: list[dict[str, str]] = []

        if source == "jsonl":
            path = PROMPTS_DIR / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run: python -m src.phase2.prompts --export-all"
                )
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.rows.append(
                        {
                            "db_id": rec["db_id"],
                            "question": rec["question"],
                            "query": rec["query"],
                        }
                    )
        elif source == "parquet":
            import pandas as pd

            path = HF_EXPORT_DIR / f"{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run Phase 1 download.")
            df = pd.read_parquet(path, columns=["db_id", "question", "query"])
            for row in df.itertuples(index=False):
                self.rows.append(
                    {"db_id": row.db_id, "question": row.question, "query": row.query}
                )
        else:
            raise ValueError("source must be 'parquet' or 'jsonl'")

        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        schema = get_schema(row["db_id"])
        ex = build_masked_example(
            self.tokenizer,
            schema=schema,
            question=row["question"],
            sql=row["query"],
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def collate_pad(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pad a list of examples to the longest sequence in the batch."""
    max_len = max(x["input_ids"].size(0) for x in batch)

    def pad_1d(t: torch.Tensor, value: int) -> torch.Tensor:
        if t.size(0) == max_len:
            return t
        out = torch.full((max_len,), value, dtype=t.dtype)
        out[: t.size(0)] = t
        return out

    return {
        "input_ids": torch.stack([pad_1d(x["input_ids"], pad_token_id) for x in batch]),
        "attention_mask": torch.stack([pad_1d(x["attention_mask"], 0) for x in batch]),
        "labels": torch.stack([pad_1d(x["labels"], IGNORE_INDEX) for x in batch]),
    }


def _print_example_report(tokenizer, ex: dict[str, Any], *, db_id: str = "") -> None:
    table = Table(title="Masked example")
    table.add_column("Field")
    table.add_column("Value")
    if db_id:
        table.add_row("db_id", db_id)
    table.add_row("cut method", ex["method"])
    table.add_row("total tokens", str(ex["n_tokens"]))
    table.add_row("masked (prompt) tokens", str(ex["n_prompt"]))
    table.add_row("supervised (SQL) tokens", str(ex["n_supervised"]))
    table.add_row(
        "mask ratio",
        f"{ex['n_prompt'] / max(ex['n_tokens'], 1):.1%} of sequence ignored in loss",
    )
    console.print(table)

    supervised_ids = ex["input_ids"][ex["cut"] :]
    decoded = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    console.print("\n[bold]Supervised span (decoded)[/bold] — model learns only this:")
    console.print(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build / inspect Text-to-SQL examples with -100 loss masking"
    )
    parser.add_argument("--model-id", default=None, help="Tokenizer Hub id")
    parser.add_argument(
        "--from-parquet",
        metavar="SPLIT",
        choices=["train", "validation"],
        help="Smoke-test one row from parquet",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--dataset",
        metavar="SPLIT",
        choices=["train", "validation"],
        help="Load Text2SqlDataset and print size + first example stats",
    )
    parser.add_argument(
        "--source",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Row source for --dataset (default: parquet)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset rows")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_id)

    if args.from_parquet:
        import pandas as pd

        path = HF_EXPORT_DIR / f"{args.from_parquet}.parquet"
        df = pd.read_parquet(path)
        row = df.iloc[args.index]
        schema = get_schema(row["db_id"])
        ex = build_masked_example(
            tokenizer,
            schema=schema,
            question=row["question"],
            sql=row["query"],
            max_length=args.max_length,
        )
        console.print(
            f"[bold]Row[/bold] {args.from_parquet}[{args.index}]  "
            f"Q: {row['question']}"
        )
        _print_example_report(tokenizer, ex, db_id=row["db_id"])
        return

    if args.dataset:
        ds = Text2SqlDataset(
            tokenizer,
            split=args.dataset,
            source=args.source,
            max_length=args.max_length,
            limit=args.limit or 8,
        )
        console.print(f"[bold]Dataset[/bold] {args.dataset}: showing {len(ds)} rows (limit)")
        sample = ds[0]
        n_mask = int((sample["labels"] == IGNORE_INDEX).sum())
        n_sup = int((sample["labels"] != IGNORE_INDEX).sum())
        console.print(
            f"example[0] shapes: input_ids={tuple(sample['input_ids'].shape)}  "
            f"masked={n_mask}  supervised={n_sup}"
        )
        # Rebuild rich report for row 0
        row = ds.rows[0]
        schema = get_schema(row["db_id"])
        ex = build_masked_example(
            tokenizer,
            schema=schema,
            question=row["question"],
            sql=row["query"],
            max_length=args.max_length,
        )
        _print_example_report(tokenizer, ex, db_id=row["db_id"])
        return

    parser.print_help()
    console.print(
        "\n[bold]Try:[/bold]\n"
        "  python -m src.phase3.dataset --from-parquet train --index 0\n"
        "  python -m src.phase3.dataset --dataset train --limit 4\n"
    )


if __name__ == "__main__":
    main()
