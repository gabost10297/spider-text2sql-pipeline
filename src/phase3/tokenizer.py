"""Llama-3 Instruct tokenizer — load, inspect, and round-trip text.

WHY THIS EXISTS
---------------
An LLM never sees raw strings. A *tokenizer* maps text <> integer IDs:

    "SELECT count(*)"  →  [token_id, token_id, ...]  →  model

Those IDs are looked up in an embedding matrix. Wrong tokenizer = wrong IDs =
garbage generations, even if the base model weights are correct.

For Llama-3 Instruct we must use the tokenizer that matches the checkpoint
(same vocab + same special tokens like ``<|eot_id|>``). Our Phase 2 prompts
were written in that chat format on purpose.

WHAT A TOKENIZER ACTUALLY DOES
------------------------------
1. **Normalise** text (unicode, spaces — mostly handled for you).
2. **Split** into subword pieces (BPE / SentencePiece-style). Long rare words
   become several tokens; common words are often one token.
3. **Map** each piece to a vocabulary integer (``input_ids``).
4. Optionally add **special tokens** (BOS, EOS, padding).

Decode reverses the map: IDs → readable text (with a few edge cases around
spaces and special tokens).

SPECIAL TOKENS IN OUR PROMPTS
-----------------------------
Llama-3 Instruct chat uses markers our Phase 2 strings already embed:

- ``<|begin_of_text|>``     — start of sequence (BOS)
- ``<|start_header_id|>`` / ``<|end_header_id|>`` — role headers (system/user/assistant)
- ``<|eot_id|>``            — end of turn

The tokenizer must treat these as *atomic* special tokens (one ID each), not
as character soup. That is why we load the official Instruct tokenizer.

PAD TOKEN
---------
Llama-3 often has no dedicated pad token. For batching later we usually set
``pad_token = eos_token`` (or ``eot``). We do that here so Phase 3 Dataset
code can pad safely.

THIS MODULE DOES NOT
--------------------
- Download model *weights* (only tokenizer files).
- Apply loss masking (next step: Dataset).
- Fine-tune anything.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from rich.console import Console
from rich.table import Table

# Default: Unsloth Instruct (ungated tokenizer). Override with $TOKENIZER_ID
# or use meta-llama/Meta-Llama-3-8B-Instruct + HF_TOKEN after accepting the license.
DEFAULT_TOKENIZER_ID = os.environ.get(
    "TOKENIZER_ID",
    "unsloth/llama-3-8b-Instruct",
)

console = Console()


def load_tokenizer(model_id: str | None = None, *, trust_remote_code: bool = False):
    """Download/load the tokenizer and return a Hugging Face ``PreTrainedTokenizer``.

    Parameters
    ----------
    model_id:
        Hub repo that ships the tokenizer files. Must match the chat template
        of the model you will fine-tune.
    trust_remote_code:
        Leave False unless a repo explicitly needs custom tokenizer code.
    """
    # Import here so `python -m ... --help` works even before transformers is installed.
    from transformers import AutoTokenizer

    repo = model_id or DEFAULT_TOKENIZER_ID
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    console.print(f"[bold]Loading tokenizer[/bold]: {repo}")
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    # Llama-3 Instruct typically has no pad_token. Batching needs one.
    if tokenizer.pad_token is None:
        # Prefer eos; many Llama-3 setups use eot as the conversational end.
        tokenizer.pad_token = tokenizer.eos_token
        console.print(
            f"[dim]pad_token was unset → using eos_token "
            f"{tokenizer.eos_token!r} (id={tokenizer.pad_token_id})[/dim]"
        )

    return tokenizer


def describe_tokenizer(tokenizer) -> None:
    """Print vocab size and the special tokens we care about."""
    table = Table(title="Tokenizer overview")
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("class", type(tokenizer).__name__)
    table.add_row("vocab_size", f"{tokenizer.vocab_size:,}")
    table.add_row("model_max_length", str(tokenizer.model_max_length))
    table.add_row("bos_token", repr(tokenizer.bos_token))
    table.add_row("eos_token", repr(tokenizer.eos_token))
    table.add_row("pad_token", repr(tokenizer.pad_token))
    table.add_row("unk_token", repr(getattr(tokenizer, "unk_token", None)))

    # Chat-related specials (may appear under additional_special_tokens)
    interesting = (
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
    )
    for name in interesting:
        tid = tokenizer.convert_tokens_to_ids(name)
        # HF returns unk id if missing — check string round-trip
        known = tokenizer.convert_ids_to_tokens(tid) == name or name in (
            tokenizer.bos_token,
            tokenizer.eos_token,
        )
        table.add_row(f"id({name})", str(tid) if known else "NOT IN VOCAB")

    console.print(table)


def encode_text(
    tokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Encode *text* → token strings + ids.

    ``add_special_tokens=False`` is important for our Phase 2 prompts: they
    already contain ``<|begin_of_text|>`` etc. If we leave the default True,
    the tokenizer may *double-prepend* BOS and distort the sequence.
    """
    # encode() returns List[int]; tokenize() returns List[str] pieces.
    token_strs = tokenizer.tokenize(text)
    # convert_tokens_to_ids does not add BOS/EOS by itself
    token_ids = tokenizer.convert_tokens_to_ids(token_strs)

    if add_special_tokens:
        # Rare path — shows what HF would add if you asked it to.
        encoded = tokenizer(text, add_special_tokens=True)
        token_ids = encoded["input_ids"]
        token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "token_strs": token_strs,
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
    }


def decode_ids(tokenizer, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
    """IDs → string. Set skip_special_tokens=True to hide chat markers."""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def show_token_table(
    tokenizer,
    text: str,
    *,
    max_rows: int = 40,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    """Pretty-print the first tokens and verify round-trip decode."""
    encoded = encode_text(tokenizer, text, add_special_tokens=add_special_tokens)
    ids = encoded["token_ids"]
    strs = encoded["token_strs"]

    table = Table(title=f"Tokens (showing {min(len(ids), max_rows)} / {len(ids)})")
    table.add_column("#", justify="right")
    table.add_column("token")
    table.add_column("id", justify="right")

    for i, (tok, tid) in enumerate(zip(strs, ids)):
        if i >= max_rows:
            table.add_row("…", f"({len(ids) - max_rows} more)", "…")
            break
        # Ġ / space markers are easier to see with repr
        table.add_row(str(i), repr(tok), str(tid))

    console.print(table)
    console.print(f"[bold]Total tokens:[/bold] {encoded['n_tokens']}")

    roundtrip = decode_ids(tokenizer, ids, skip_special_tokens=False)
    console.print("\n[bold]Round-trip decode[/bold] (special tokens kept):")
    console.print(roundtrip[:2000] + ("…" if len(roundtrip) > 2000 else ""))

    if roundtrip != text:
        console.print(
            "\n[yellow]Note:[/yellow] decode ≠ original byte-for-byte "
            "(normal for BPE spacing / special-token rendering). "
            "Semantic content should still match."
        )
    else:
        console.print("\n[green]Round-trip matches original text exactly.[/green]")

    return encoded


def find_sql_span_hint(tokenizer, prompt: str) -> None:
    """Teaching helper: locate the 'SQL: ' marker in token space.

    Loss masking (next step) will keep labels only *after* this marker's
    content — i.e. on the SQL tokens themselves. Here we only *show* where
    that boundary sits; we do not build labels yet.
    """
    marker = "SQL: "
    if marker not in prompt:
        console.print("[yellow]No 'SQL: ' marker in text — skip span hint.[/yellow]")
        return

    prefix, _, rest = prompt.partition(marker)
    # Tokens for everything through 'SQL: ' (inclusive of marker text)
    prefix_and_marker = prefix + marker
    n_prefix = encode_text(tokenizer, prefix_and_marker)["n_tokens"]
    n_total = encode_text(tokenizer, prompt)["n_tokens"]
    n_sql = n_total - n_prefix

    console.print(
        f"\n[bold]Masking preview (conceptual)[/bold]\n"
        f"  tokens through 'SQL: '  → indices [0 .. {n_prefix - 1}]  "
        f"(will become labels=-100 later)\n"
        f"  SQL / answer tokens     → indices [{n_prefix} .. {n_total - 1}]  "
        f"({n_sql} tokens keep real label ids)\n"
        f"  First SQL snippet: {rest[:80]!r}…"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and inspect the Llama-3 Instruct tokenizer"
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"HF repo id (default: {DEFAULT_TOKENIZER_ID} or $TOKENIZER_ID)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Tokenize a tiny built-in SQL snippet (no Spider data needed)",
    )
    parser.add_argument(
        "--from-parquet",
        metavar="SPLIT",
        choices=["train", "validation"],
        help="Build a Phase 2 training prompt from parquet and tokenize it",
    )
    parser.add_argument("--index", type=int, default=0, help="Parquet row index")
    parser.add_argument(
        "--text",
        help="Tokenize this literal string",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Let HF prepend BOS/etc. (usually OFF for our pre-formatted prompts)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=40,
        help="Max rows in the token table",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_id)
    describe_tokenizer(tokenizer)

    text: str | None = None
    if args.demo:
        text = (
            "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "SQL: SELECT count(*) FROM singer<|eot_id|>"
        )
    elif args.text:
        text = args.text
    elif args.from_parquet:
        from src.phase2.prompts import generate_prompt_for_example
        import pandas as pd
        from src.phase1.paths import HF_EXPORT_DIR

        path = HF_EXPORT_DIR / f"{args.from_parquet}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing {path} — run Phase 1 download first.")
        df = pd.read_parquet(path)
        row = df.iloc[args.index]
        text = generate_prompt_for_example(row["db_id"], row["question"], row["query"])
        console.print(
            f"\n[bold]Example[/bold] {args.from_parquet}[{args.index}] "
            f"db_id={row['db_id']}"
        )
    else:
        parser.print_help()
        console.print(
            "\n[bold]Try:[/bold]\n"
            "  python -m src.phase3.tokenizer --demo\n"
            "  python -m src.phase3.tokenizer --from-parquet train --index 0\n"
        )
        return

    show_token_table(
        tokenizer,
        text,
        max_rows=args.max_rows,
        add_special_tokens=args.add_special_tokens,
    )
    find_sql_span_hint(tokenizer, text)


if __name__ == "__main__":
    main()
