# RAG Against The Machine

## 1. System Architecture

The RAG system follows a classic pipeline with four main stages:

1. **Ingestion (`DataIngester`)**: Traverses `data/raw`, filters relevant files by extension, applies language-specific chunking, and normalizes text into tokens.
2. **Indexing (`Indexer`)**: Builds separate BM25 indices for documentation and code. Doc chunks and code chunks are tokenized and stored with metadata (`MinimalSource`).
3. **Retrieval (`Retriever`)**: Accepts a user query, normalizes it, and queries the BM25 index. Supports single-index (`docs`/`code`) or combined interleaved retrieval.
4. **Generation (`AnswerGenerator`)**: Loads Qwen/Qwen3-0.6B, formats a prompt with retrieved context, and generates a constrained answer.

## 2. Chunking Strategy

Two strategies are implemented:

- **Documentation / Markdown**: Chunks are split at paragraph boundaries (`\n\n`) when possible, falling back to line breaks (`\n`). Maximum size is configurable (default 2000 chars). Overlap is 250 chars.
- **Code (Python/C++/CUDA/Shell)**: Chunks are split at structural boundaries: function definitions (`\ndef `), class definitions (`\nclass `), async functions, or comment blocks. Maximum size is 1500 chars with 300 chars overlap to preserve context across logical units.

**Impact of chunk size**: If chunks are too small, context is fragmented and retrieval misses broader context. If too large, BM25 precision drops because the token distribution becomes noisy and the relevant signal is diluted.

## 3. Retrieval Method

We use **BM25** via the `bm25s` library. BM25 is a bag-of-words ranking function that scores documents based on term frequency and inverse document frequency, with saturation and length normalization.

**Ranking mechanism**: For each query token, BM25 computes a score per chunk. The top-k chunks with the highest aggregated scores are returned. For combined search, we interleave results from the docs and code indices to provide a balanced context.

## 4. Performance Analysis

Metrics on the private evaluation datasets:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| Docs    | 67.0%    | 79.0%    | **81.0%** | 83.0%     |
| Code    | 39.0%    | 56.0%    | **63.0%** | 73.0%     |

- **Docs Recall@5**: 81.0% (threshold: 80%) ✅
- **Code Recall@5**: 63.0% (threshold: 50%) ✅

**System performance**:
- Indexing: ~45 seconds for the full vLLM codebase.
- Warm retrieval throughput: 200 questions in ~25 seconds.

## 5. Design Decisions and Trade-offs

- **BM25 over TF-IDF**: BM25 provides better length normalization and term saturation, which is critical for mixed-length code and doc chunks.
- **Separate indices for docs and code**: Prevents code syntax tokens from polluting doc retrieval and vice versa. The trade-off is slightly more memory usage.
- **Path token boosting for code**: File path tokens are repeated 3× in the code corpus to boost structural relevance (e.g., a file named `paged_attention.py` is more likely to match queries about PagedAttention).
- **Qwen3-0.6B as default**: Small enough to run on CPU if needed, yet capable of generating coherent answers. The trade-off is limited reasoning depth compared to larger models.
- **No query expansion / embeddings**: Kept the system deterministic and fast. Adding semantic embeddings or query rewriting could improve recall further.

## 6. Example Usage

```bash
# Install dependencies
uv sync

# Index the vLLM codebase
uv run python -m src index --max_chunk_size 2000

# Search for relevant chunks
uv run python -m src search "How to configure OpenAI server?" --k 10

# Generate an answer
uv run python -m src answer "What is PagedAttention?" --k 10

# Batch search over a dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/private/UnansweredQuestions/dataset_docs_private.json \
    --k 10 \
    --save_directory data/output/search_results

# Evaluate retrieval performance
uv run python -m src evaluate \
    --student_answer_path data/output/search_results/results_dataset_docs_private.json \
    --dataset_path data/datasets/private/AnsweredQuestions/dataset_docs_private.json
```
