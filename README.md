# arXiv Semantic Search & RAG Pipeline

**Databricks + Apache Spark + Delta Lake + FAISS + HuggingFace**

This project builds an end-to-end **semantic search and Retrieval-Augmented Generation (RAG)** system** on the arXiv metadata dataset using Apache Spark on Databricks.

The pipeline processes ~5GB of raw JSON, generates embeddings for document chunks, builds a FAISS vector index, and enables LLM-powered question answering over arXiv papers.

Dataset source:
[https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
Provided by **Cornell University**

---

# 📌 Architecture

This project follows a **Medallion Architecture**:

```
Raw JSON (5GB)
    ↓
Bronze (Raw Delta)
    ↓
Silver (Cleaned & Normalized)
    ↓
Gold (Chunked + Embedded)
    ↓
FAISS Vector Index
    ↓
LLM (RAG)
```

Designed to run on **Databricks with Spark**.

---

# 🥉 Bronze Layer — Raw Ingestion

### Goal

Ingest the full 5GB arXiv JSON snapshot into Delta Lake.

### What Happens

* Read JSON using Spark
* Enable permissive parsing
* Add ingestion timestamp
* Write to Delta format

### Output

Delta table:

```
arxiv_bronze
```

Storage location:

```
/FileStore/delta/arxiv_bronze
```

This layer preserves the original dataset in an ACID-compliant format.

---

# 🥈 Silver Layer — Cleaning & Normalization

### Goal

Prepare structured, clean data for embedding.

### Transformations

* Select relevant columns:

  * id
  * title
  * abstract
  * authors
  * categories
  * versions
* Lowercase & trim text fields
* Remove null abstracts
* Add clean processing timestamp

### Output

Delta table:

```
arxiv_silver
```

Storage location:

```
/FileStore/delta/arxiv_silver
```

This layer contains clean, normalized metadata ready for NLP processing.

---

# 🥇 Gold Layer — Chunking + Embeddings

This is where semantic search becomes possible.

---

## 🔹 Step 1: Chunking (Spark)

Each abstract is:

1. Split into sentences
2. Grouped into chunks of 5 sentences
3. Recombined into `chunk_text`

Why chunk?

* Improves retrieval granularity
* Avoids long-context embedding degradation
* Better for RAG context windows

Resulting columns:

| Column     | Description             |
| ---------- | ----------------------- |
| id         | arXiv paper ID          |
| title      | Cleaned title           |
| categories | arXiv categories        |
| chunk_id   | Chunk index             |
| chunk_text | Text used for embedding |

---

## 🔹 Step 2: Creating Embeddings

Embeddings are generated using:

**Hugging Face SentenceTransformers**

Model used:
**all-MiniLM-L6-v2**

### How it works

* Each `chunk_text` is encoded into a 384-dimensional dense vector
* Batched (5,000 rows per batch)
* Computed on the driver node
* Stored as `array<float>` in Spark

Example:

```
embedding: array<float> (dim = 384)
```

---

## 🔹 Step 3: Gold Storage

Delta table:

```
arxiv_gold
```

Storage location:

```
/FileStore/delta/arxiv_gold
```

Each row contains both the chunk text and its embedding vector.

Embeddings are persisted — they are **not recomputed at query time**.

---

# 🔎 Vector Search with FAISS

For fast similarity search, embeddings are loaded into:

**FAISS**

### Index Type

```
IndexFlatIP
```

Why?

* Vectors are L2-normalized
* Inner product becomes cosine similarity

### Retrieval Flow

1. Convert Spark → Pandas
2. Stack embeddings into NumPy matrix
3. Normalize vectors
4. Add to FAISS index
5. Query top-k nearest neighbors

This enables millisecond-level semantic search.

---

# 🤖 Retrieval-Augmented Generation (RAG)

LLM used:
**phi-2**

### Process

1. Embed user query
2. Retrieve top-k similar chunks from FAISS
3. Build constrained prompt using retrieved context
4. Generate answer using LLM

Prompt enforces:

> If the answer is not in the context, respond that the context does not contain the answer.

This reduces hallucinations and ensures grounded responses.

---

# 📦 Data Storage Strategy

All processing layers are stored as **Delta Lake tables**.

| Layer  | Format | Purpose                   |
| ------ | ------ | ------------------------- |
| Bronze | Delta  | Raw ingestion             |
| Silver | Delta  | Cleaned metadata          |
| Gold   | Delta  | Chunked text + embeddings |

Benefits:

* ACID transactions
* Schema enforcement
* Time travel support
* Scalable Spark queries

Embeddings are stored as `array<float>` inside the Gold Delta table.

---

# 🚀 Running on Databricks

This notebook is designed for:

* Databricks Runtime
* Spark cluster
* Driver node with sufficient memory for embedding & FAISS

Recommended:

* Large driver memory (for embedding stage)
* Optional GPU for faster embedding

---

# 📈 Scalability Notes

Current design:

* Embeddings computed on driver
* FAISS index built in-memory

Future improvements:

* Distributed embedding using Spark UDFs
* IVF or HNSW FAISS index for larger scale
* Persist FAISS index to storage
* Add metadata-based filtering
* Use instruction-tuned LLM for better answers

---

# 🧠 What This Project Demonstrates

* Large-scale JSON ingestion with Spark
* Medallion architecture (Bronze/Silver/Gold)
* Text chunking strategies for NLP
* Dense vector embedding generation
* Delta Lake storage design
* FAISS vector search
* End-to-end Retrieval-Augmented Generation

---

# 📌 Example Use Case

Query:

```
How are graph neural networks used for molecular property prediction?
```

System will:

* Embed the query
* Retrieve relevant arXiv chunks
* Generate a grounded answer using retrieved context

---

# 📄 License & Dataset

Dataset provided by **Cornell University arXiv** via Kaggle.

Please review dataset licensing on Kaggle before redistribution.

---

# ⭐ Summary

This project transforms raw arXiv metadata into a production-style semantic search and RAG system using:

* Apache Spark
* Delta Lake
* SentenceTransformers
* FAISS
* HuggingFace LLMs

It demonstrates how to build scalable AI search systems on Databricks from raw data to LLM-powered answers.
