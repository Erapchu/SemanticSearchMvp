# eFiler Semantic Search MVP

Minimal .NET 10 console prototype showing three things side-by-side:

1. existing-style full-text search with Lucene.NET;
2. local semantic search with `intfloat/multilingual-e5-small` via ONNX Runtime;
3. a tiny hybrid ranker using Reciprocal Rank Fusion (RRF).

The four example emails are hard-coded in `Program.cs`.

## Run

```powershell
./download-model.ps1
dotnet restore
dotnet run
```

The script downloads the official optimized ONNX model plus the Hugging Face `tokenizer.json` into:

```text
models/multilingual-e5-small/
```

The model is deliberately **not** included in this repository/zip because it is roughly 235 MB.

## What to look for

The first query is Russian:

```text
тортики на заказ
```

The relevant email is English and contains:

```text
we make bespoke birthday cakes to order
```

Lucene has no lexical bridge between those phrases, while multilingual E5 should rank that email highly by meaning.

## Architecture represented by the MVP

```text
Parsed email
   |---> Lucene fields (subject/from/to/body/...)
   `---> "passage: ..." ---> multilingual-e5-small ---> float[384]

User query
   |---> Lucene query
   `---> "query: ..." ---> multilingual-e5-small ---> float[384]
                                      |
                                      `--> cosine similarity
```

`HybridSearch.cs` merges both result lists using RRF rather than comparing BM25 and cosine scores directly.

## Production changes for eFiler

This sample intentionally keeps vectors in RAM and embeds one vector per email. For production:

- chunk long email bodies (E5 max context is 512 tokens);
- persist vectors keyed by the same stable eFiler/Lucene document id;
- batch embedding generation during indexing;
- store an embedding model/version with each vector so a model upgrade can trigger reindexing;
- use ANN/HNSW once brute-force cosine becomes too slow;
- keep Lucene metadata filters and exact search — semantic search complements it rather than replacing it;
- RAG is the next layer: take top chunks from hybrid search and send only those chunks to the selected LLM, asking it to answer with source document IDs.

## Files

- `LuceneMailIndex.cs` — RAM Lucene index and lexical search.
- `E5EmbeddingService.cs` — tokenizer + ONNX inference + E5 average pooling + L2 normalization.
- `SemanticMailIndex.cs` — in-memory vectors and cosine search.
- `HybridSearch.cs` — simple RRF result fusion.
- `Program.cs` — four fake emails and four example queries.

