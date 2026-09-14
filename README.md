# Text-to-SQL RAG Assistant with Multi-Model Evaluation

A Retrieval-Augmented Generation (RAG) system that answers natural language questions over a relational database by generating schema-grounded SQL queries via an LLM, executing them, and producing natural language answers from the retrieved data — with multi-turn conversation support and a full evaluation suite benchmarking multiple open-source LLMs.

## Overview

This project implements an end-to-end pipeline that lets a user ask questions in plain English about data stored in a relational database (MySQL) and get accurate, data-grounded answers back - without writing any SQL themselves. Rather than relying on a vector-based document retriever, retrieval here means: **generate SQL → execute it → use the real query result as the retrieved context** for the LLM's final answer.

The project also includes a full evaluation framework to benchmark multiple open-source LLMs against each other on this task.

## Key Features

- **Text-to-SQL generation** - an LLM translates natural language questions into SQL, grounded in a dynamically-generated schema description (tables, columns, primary keys, foreign key relationships) so it never hallucinates non-existent columns or joins.
- **SQL-based retrieval** - instead of vector similarity search, retrieval is the actual SQL query result, converted into a structured text context for the LLM.
- **Multi-turn conversation support** - follow-up questions ("What about conversions?") are automatically condensed into standalone questions using chat history before SQL generation.
- **Data cleaning pipeline** - missing value imputation (median/categorical), IQR-based outlier detection, and PK/FK integrity validation, run on the raw MySQL tables before they're used for querying.
- **Constraint-preserving in-memory layer** - cleaned data is loaded into a fresh in-memory database that recreates the original schema's primary and foreign key constraints, so schema introspection stays accurate without modifying the live source database.
- **Multi-model benchmarking** - the entire pipeline runs identically across multiple open-source LLMs (TinyLlama, Llama 3.2, Gemma2:2b) for direct, apples-to-apples comparison.
- **Comprehensive evaluation suite** - 9 metrics across two frameworks:
  - **RAGAS:** Faithfulness, Context Recall, Factual Correctness
  - **NLG metrics:** BLEU, METEOR, ROUGE, Intrinsic Perplexity, BERTScore, BARTScore

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Database | MySQL, SQLite (in-memory) |
| ORM / DB Access | SQLAlchemy, Pandas |
| LLM Orchestration | LangChain |
| LLM Runtime | Ollama (TinyLlama, Llama 3.2, Gemma2:2b) |
| Evaluation | RAGAS, NLTK, BERTScore, Hugging Face Transformers (GPT-2, BART) |
| Deep Learning Backend | PyTorch |
| Visualization | Matplotlib |

## Project Structure

```
├── data_cleaning.py        # MySQL connection, data cleaning, constraint-preserving
│                            # in-memory engine setup
├── rag_common.py            # Shared pipeline: schema description, Text-to-SQL,
│                            # retrieval, answer generation, chat history handling,
│                            # and all evaluation metric functions
├── run_model.py              # Runs the full pipeline + evaluation for a single model
│                            # (run once per model, saves results to a pickle file)
└── consolidate_results.py    # Merges all per-model results into a unified CSV
                             # with comparison summary tables and charts
```

## How It Works

1. **`data_cleaning.py`** connects to the source MySQL database, loads all tables into Pandas, cleans them (missing values, outlier detection), and rebuilds the cleaned data into a fresh in-memory database — carrying over the original primary and foreign key constraints so relationships stay intact.
2. **`rag_common.py`** introspects that cleaned database to build a schema description (tables, columns, and foreign key relationships), which is injected into every SQL-generation prompt to keep the LLM grounded in the real schema.
3. For each question:
   - If there's prior conversation history, the question is first condensed into a standalone question.
   - The LLM generates a SQL query using the schema-grounded prompt.
   - The query is executed against the cleaned database.
   - The result is converted into a structured text context.
   - The LLM generates a final natural language answer using that context.
4. **`run_model.py`** runs this full pipeline against a fixed set of test questions for one model at a time, then scores every response using RAGAS and the six NLG metrics.
5. **`consolidate_results.py`** merges the results from all three models into a single CSV and generates comparison bar charts, enabling side-by-side evaluation of model performance across every metric.

## Setup

```bash
pip install pandas numpy sqlalchemy pymysql langchain langchain-ollama \
            ragas nltk rouge bert-score transformers torch matplotlib
```

Update the MySQL connection string in `data_cleaning.py` with your own credentials:

```python
mysql_engine = create_engine("mysql+pymysql://<user>:<password>@<host>/<database>")
```

Pull the models you want to benchmark via [Ollama](https://ollama.com):

```bash
ollama pull tinyllama
ollama pull llama3.2
ollama pull gemma2:2b
```

## Running the Pipeline

```bash
# 1. Run once per model (edit MODEL_NAME / MODEL_ID at the top of run_model.py)
python run_model.py

# 2. After all three models have been run, consolidate results
python consolidate_results.py
```

Outputs:
- `<model>_scores.pkl` - per-model results and metric scores
- `model_comparison_results.csv` - merged per-question, per-model comparison
- `model_comparison_metrics\perplexity.png` - model output in chart form

## Evaluation Approach

Each model is tested on an identical set of questions to ensure a fair comparison. Two complementary evaluation angles are used:

- **RAGAS** evaluates the RAG system's behavior specifically - whether answers are faithful to the retrieved data, whether the right context was retrieved, and factual correctness against a reference answer.
- **NLG metrics** (BLEU, METEOR, ROUGE, Perplexity, BERTScore, BARTScore) evaluate answer quality independently, across lexical overlap, semantic similarity, and fluency - providing a broader picture than any single metric alone.


