# =========================================================
# rag_common.py
# Shared pipeline + evaluation functions, used by every model run.
# `engine` is imported from data_cleaning.py (cleaned + constrained
# in-memory DB) — this file does not connect to MySQL itself.
# =========================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from sqlalchemy import inspect
from langchain_core.prompts import PromptTemplate

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BartForConditionalGeneration, BartTokenizer
from bert_score import BERTScorer

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

import pickle

# ---------------------------------------------------------
# SECTION 1: Setup — engine (from data_cleaning.py) + schema
# ---------------------------------------------------------
from data_cleaning import engine

# llm = your existing LLM object (OllamaLLM, etc.) — passed into
# the functions below by run_model.py, one model at a time.


def build_schema_description(engine):
    inspector = inspect(engine)
    lines = []
    for table_name in inspector.get_table_names():
        pk_info = inspector.get_pk_constraint(table_name)
        pk_cols = pk_info.get("constrained_columns", [])
        pk = pk_cols[0] if pk_cols else "none"
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        lines.append(f"Table: {table_name}\nPrimary Key: {pk}\nColumns: {', '.join(columns)}")

    lines.append("\nForeign Key Relationships:")
    for table_name in inspector.get_table_names():
        for fk in inspector.get_foreign_keys(table_name):
            child_col = fk["constrained_columns"][0]
            parent_table = fk["referred_table"]
            parent_col = fk["referred_columns"][0]
            lines.append(f"- {table_name}.{child_col} references {parent_table}.{parent_col}")

    return "\n\n".join(lines)


schema = build_schema_description(engine)


# ---------------------------------------------------------
# SECTION 2: Standalone question condensation (chat history aware)
# ---------------------------------------------------------
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that makes sense without the chat history.
If the follow-up question is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow-up Input: {question}
Standalone question:""")


def format_chat_history(chat_history):
    """chat_history is a list of (question, answer) tuples."""
    if not chat_history:
        return "No previous conversation."
    lines = []
    for q, a in chat_history:
        lines.append(f"User: {q}\nAssistant: {a}")
    return "\n".join(lines)


def _extract_text(response):
    return response.content if hasattr(response, "content") else response


def condense_question(llm, question, chat_history):
    if not chat_history:
        return question
    prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=format_chat_history(chat_history),
        question=question
    )
    return _extract_text(llm.invoke(prompt)).strip()


# ---------------------------------------------------------
# SECTION 3: SQL generation (schema-grounded)
# ---------------------------------------------------------
sql_prompt = """You are a SQL expert. Use ONLY the tables, columns, and foreign key relationships listed below.
Never assume a column exists in a table unless it is explicitly listed under that table.
If the requested information spans multiple tables, JOIN them using the foreign key relationships provided.

Schema:
{schema}

Question: {question}

Return only the raw SQL query. No markdown, no explanation.
"""


def generate_sql(llm, question):
    prompt = sql_prompt.format(schema=schema, question=question)
    sql_query = _extract_text(llm.invoke(prompt))
    return sql_query.strip().replace("```sql", "").replace("```", "").strip()


# ---------------------------------------------------------
# SECTION 4: Retrieved context (from SQL result)
# ---------------------------------------------------------
def build_retrieved_context(result_df, source_name="MySQL Database"):
    """
    Converts the SQL query result into one context string.
    """
    if result_df.empty:
        return f"Source: {source_name}\n\nNo relevant data was found."

    table_data = result_df.to_string(index=False)
    retrieved_context = f"Source: {source_name}\n\nRelevant Data:\n{table_data}"
    return retrieved_context


# ---------------------------------------------------------
# SECTION 5: Answer generation (from retrieved context)
# ---------------------------------------------------------
answer_prompt = """You are a helpful data analyst. Answer the user's question using ONLY the data provided below.
If the data does not contain enough information to answer, say so clearly.

Question: {question}

Retrieved Data:
{context}

Answer:"""


def generate_answer(llm, question, context):
    prompt = answer_prompt.format(question=question, context=context)
    return _extract_text(llm.invoke(prompt)).strip()


# ---------------------------------------------------------
# SECTION 6: End-to-end query processor
# ---------------------------------------------------------
def process_query(llm, question, chat_history):
    """
    Runs the full pipeline: condense -> generate SQL -> execute -> build context -> generate answer.
    Returns (answer, retrieved_context, standalone_question, sql_query, result_df)
    """
    standalone_question = condense_question(llm, question, chat_history)

    sql_query = generate_sql(llm, standalone_question)

    try:
        result_df = pd.read_sql(sql_query, engine)
    except Exception as e:
        result_df = pd.DataFrame()
        print(f"SQL execution failed: {e}\nQuery was:\n{sql_query}")

    retrieved_context = build_retrieved_context(result_df, source_name="MySQL Database")
    answer = generate_answer(llm, standalone_question, retrieved_context)

    return answer, retrieved_context, standalone_question, sql_query, result_df


def run_pipeline_on_queries(llm, sample_queries):
    """Runs the full RAG pipeline for one model across all sample questions."""
    results = []
    chat_history = []
    for item in sample_queries:
        question = item["question"]
        ground_truth = item["answer"]
        answer, retrieved_context, _, _, _ = process_query(llm, question, chat_history)
        results.append({
            "user_input": question,
            "reference": ground_truth,
            "response": answer,
            "retrieved_contexts": [retrieved_context]
        })
        chat_history.append((question, answer))
    return results


# ---------------------------------------------------------
# SECTION 7: Interactive chat loop (maintains chat history)
# ---------------------------------------------------------
def make_ask_function(llm):
    """Returns an `ask(question)` closure bound to a specific model + its own chat history."""
    chat_history = []

    def ask(question):
        answer, context, standalone_q, sql_q, result_df = process_query(llm, question, chat_history)
        print("Standalone question:", standalone_q)
        print("\nGenerated SQL:", sql_q)
        print("\nSQL Result:\n", result_df)
        print("\nAnswer:", answer)
        chat_history.append((question, answer))
        return answer

    return ask

# Example:
# ask = make_ask_function(llm)
# ask("Which campaign has the highest number of clicks?")
# ask("What about conversions?")   # <- condensed to standalone using chat history


# ---------------------------------------------------------
# SECTION 9 (results builder, matches RAGAS's expected schema)
# ---------------------------------------------------------
def build_results(llm, sample_queries):
    return run_pipeline_on_queries(llm, sample_queries)


# =========================================================
# SECTION 12: Intrinsic PPL-Score
# =========================================================
_ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
_ppl_model.eval()


def calculate_perplexity(text, model=_ppl_model, tokenizer=_ppl_tokenizer):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    if input_ids.size(1) == 0:
        return float("inf")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()


def compute_perplexity_scores(results):
    return [calculate_perplexity(item["response"]) for item in results]


# =========================================================
# SECTION 13: model based metrics - BERT and BART
# =========================================================
_bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn", disable_mmap=True
)
_bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
_bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def compute_bert_bart_scores(results):
    bert_scores, bart_scores = [], []
    for item in results:
        inputs = _bart_tokenizer(item["response"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = _bart_model(**inputs).logits
        bart_scores.append(logits.mean().item())

        P, R, F1 = _bert_scorer.score([item["response"]], [item["reference"]])
        bert_scores.append(F1.numpy().mean())
    return bert_scores, bart_scores


# ---------------------------------------------------------
# n-gram metrics (BLEU / ROUGE / METEOR) — same as sections 10-11 originally
# ---------------------------------------------------------
_rouge = Rouge()


def compute_ngram_metrics(results):
    bleu, rouge_one, meteor_scores = [], [], []
    for item in results:
        ref_tokens = word_tokenize(item["reference"])
        hyp_tokens = word_tokenize(item["response"])
        bleu.append(sentence_bleu([ref_tokens], hyp_tokens))
        meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
        scores = _rouge.get_scores(item["response"], item["reference"])[0]
        rouge_one.append(scores["rouge-1"]["r"])
    return bleu, rouge_one, meteor_scores


def compute_ragas_scores(llm, results):
    evaluation_dataset = EvaluationDataset.from_list(results)
    evaluator_llm = LangchainLLMWrapper(llm)
    return evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )


def evaluate_model(llm, model_name, sample_queries):
    """Runs the RAG pipeline + every metric for a single model. Returns one dict."""
    print(f"\n=== Running pipeline for {model_name} ===")
    results = build_results(llm, sample_queries)

    bleu, rouge_one, meteor_scores = compute_ngram_metrics(results)
    ppl_scores = compute_perplexity_scores(results)
    bert_scores, bart_scores = compute_bert_bart_scores(results)
    ragas_result = compute_ragas_scores(llm, results)

    return {
        "model_name": model_name,
        "results": results,
        "bleu": bleu,
        "rouge": rouge_one,
        "meteor": meteor_scores,
        "ppl": ppl_scores,
        "bert": bert_scores,
        "bart": bart_scores,
        "ragas": ragas_result,
    }


# =========================================================
# SECTION 14: Save the model
# =========================================================
def save_scores(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {filename}")
