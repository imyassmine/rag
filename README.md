# RAG over the vLLM codebase

[![CI](https://github.com/imyassmine/rag/actions/workflows/ci.yml/badge.svg)](https://github.com/imyassmine/rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**A Retrieval-Augmented Generation pipeline that answers natural-language questions about a real
~1,900-file codebase.** AST-aware chunking, identifier-tuned TF-IDF/BM25 retrieval, grounded answers
from Qwen3-0.6B and a recall@k evaluation harness — exposed as a small CLI.

```console
$ make run ARGS='answer "How to dynamically load a LoRA adapter?" --k 3'
To dynamically load a LoRA adapter, send a POST request to the
`/v1/load_lora_adapter` endpoint with the name and path to the LoRA adapter.
```

## Results

Recall on the annotated evaluation sets (100 documentation questions, 99 code questions),
`max_chunk_size=2000`, 1,937 files:

| Retriever | R@5 docs | R@5 code | R@10 docs | R@10 code |
| --- | :---: | :---: | :---: | :---: |
| TF-IDF | 75.0% | 51.5% | 81.0% | 61.6% |
| **BM25** | **79.0%** | **73.7%** | **88.0%** | **81.8%** |

A hit means the reference passage appears in the top-k sources (same file, character-span IoU ≥ 0.05).
Identifier-aware tokenization alone lifted code recall@5 from 35.4% to 65.7%; adding file-path
indexing in BM25 brings it to 73.7%. Design details and trade-offs:
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

## Highlights

- **AST-aware chunking**: one chunk per top-level function/class (decorators included) for Python,
  one per heading for Markdown; oversized chunks are hard-split and unparsable files fall back to
  fixed-size slicing.
- **Retrieval tuned for source code**: BM25 splits `snake_case`/`camelCase` identifiers into words,
  drops stop words and indexes the file path; TF-IDF uses sublinear term frequency. The chosen method
  is persisted in the index.
- **Grounded generation, cheap**: top-k passages are re-read from source via stored offsets, then
  Qwen3-0.6B answers with `enable_thinking=False` (no reasoning trace).
- **Measured, not guessed**: recall@k evaluation against annotated datasets. CI runs lint, type checks
  and tests without PyTorch, plus a manual full-pipeline run on the real corpus.
- **Typed pipeline**: Pydantic contracts between stages; `index`, `search`, `answer` and `evaluate`
  run independently.

## Quick start

Python 3.10+ and [uv](https://docs.astral.sh/uv/). The corpus is not versioned (~38 MB):

```bash
git clone https://github.com/imyassmine/rag.git && cd rag
make install
mkdir -p data/raw
curl -sL https://github.com/vllm-project/vllm/archive/refs/tags/v0.10.1.tar.gz | tar -xz -C data/raw

# 1. Build the index (~9 s; BM25 gives the best recall — see Results)
make run ARGS="index --max_chunk_size 2000 --method bm25"
# 2. Retrieve the most relevant source snippets
make run ARGS='search "How to dynamically load a LoRA adapter?" --k 5'
# 3. Generate a grounded answer (downloads Qwen3-0.6B, ~1.5 GB, first run only)
make run ARGS='answer "How to dynamically load a LoRA adapter?" --k 3'
```

| Command | Purpose |
| --- | --- |
| `index --method tfidf\|bm25` | Chunk `data/raw/` and persist `data/processed/index.pkl` |
| `search "<question>" --k 5` | Print the top-k source locations |
| `answer "<question>" --k 5` | Retrieve, then generate a grounded answer |
| `evaluate --student_search_results_path … --dataset_path …` | Compute recall@k |
| `search_dataset` / `answer_dataset` | Run the pipeline over a whole question file |

Without `make`: `uv run python -m src <command>`. The retriever is chosen at indexing time and stored
in the index, so search commands need no extra flag.

## Architecture

```
data/raw/ ──► chunking.py ──► indexer.py ──► data/processed/index.pkl

question ──► retriever.py ──┬──► generator.py ──► grounded answer
                            └──► evaluator.py ──► recall@k
```

**Stack:** Python 3.10 · scikit-learn · rank-bm25 · PyTorch · Transformers · Qwen3-0.6B · Pydantic ·
Python Fire · uv · pytest · mypy · flake8 · GitHub Actions

**Roadmap:** semantic/hybrid retrieval (embeddings + RRF; cross-encoder reranking already
benchmarked), answer-faithfulness metrics, FastAPI + Docker serving.

---
*Built at [42 Paris](https://42.fr) by [Yassmine Abdallah (@imyassmine)](https://github.com/imyassmine).*
