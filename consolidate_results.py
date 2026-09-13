# =========================================================
# consolidate_results.py
# Loads the three per-model pickles (produced by run_model.py),
# sanity-checks alignment, merges into one CSV, and builds
# comparison summary tables + bar charts.
# =========================================================
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load all three pickle files
# ---------------------------------------------------------
with open("tinyllama_scores.pkl", "rb") as f:
    tinyllama_data = pickle.load(f)

with open("llama3.2_scores.pkl", "rb") as f:
    llama_data = pickle.load(f)

with open("gemma_scores.pkl", "rb") as f:
    gemma_data = pickle.load(f)

print("All three loaded.")
print("TinyLlama keys:", tinyllama_data.keys())
print("Llama keys:", llama_data.keys())
print("Gemma keys:", gemma_data.keys())


# ---------------------------------------------------------
# 2. Sanity checks — same question count and same question order across all three
# ---------------------------------------------------------
results = tinyllama_data["results"]  # use TinyLlama's as the reference question order
expected_len = len(results)

for model_name, data in [("TinyLlama", tinyllama_data), ("Llama3.2", llama_data), ("Gemma", gemma_data)]:
    for metric in ["bleu", "rouge", "meteor", "ppl", "bert", "bart"]:
        if len(data[metric]) != expected_len:
            print(f"MISMATCH: {model_name} - {metric} has {len(data[metric])}, expected {expected_len}")

# Confirm the questions themselves line up across notebooks, not just the count
for i in range(expected_len):
    q_tiny = tinyllama_data["results"][i]["user_input"]
    q_llama = llama_data["results"][i]["user_input"]
    q_gemma = gemma_data["results"][i]["user_input"]
    if not (q_tiny == q_llama == q_gemma):
        print(f"Question mismatch at index {i}: {q_tiny} | {q_llama} | {q_gemma}")

print("Sanity check complete.")


# ---------------------------------------------------------
# 3. Pull out per-model score lists
# ---------------------------------------------------------
tinyllama_scores = {k: tinyllama_data[k] for k in ["bleu", "rouge", "meteor", "ppl", "bert", "bart"]}
llama_scores = {k: llama_data[k] for k in ["bleu", "rouge", "meteor", "ppl", "bert", "bart"]}
gemma_scores = {k: gemma_data[k] for k in ["bleu", "rouge", "meteor", "ppl", "bert", "bart"]}

print("TinyLlama scores loaded, length check:", {k: len(v) for k, v in tinyllama_scores.items()})
print("Llama scores loaded, length check:", {k: len(v) for k, v in llama_scores.items()})
print("Gemma scores loaded, length check:", {k: len(v) for k, v in gemma_scores.items()})

for model_name, score_dict in [
    ("TinyLlama", tinyllama_scores),
    ("Llama3.2", llama_scores),
    ("Gemma", gemma_scores),
]:
    for metric_name, score_list in score_dict.items():
        if len(score_list) != expected_len:
            print(f"MISMATCH: {model_name} - {metric_name} has {len(score_list)} scores, expected {expected_len}")

print("Length check complete.")


# ---------------------------------------------------------
# 4. Build one row per question, with all three models' scores side by side
# ---------------------------------------------------------
final_rows = []

for i, item in enumerate(results):
    row = {
        "Index": i,
        "Query": item["user_input"],
        "Reference": item["reference"],

        # TinyLlama
        "TinyLlama_BLEU": tinyllama_scores["bleu"][i],
        "TinyLlama_ROUGE": tinyllama_scores["rouge"][i],
        "TinyLlama_METEOR": tinyllama_scores["meteor"][i],
        "TinyLlama_PPL": tinyllama_scores["ppl"][i],
        "TinyLlama_BERT": tinyllama_scores["bert"][i],
        "TinyLlama_BART": tinyllama_scores["bart"][i],

        # Llama 3.2
        "Llama3.2_BLEU": llama_scores["bleu"][i],
        "Llama3.2_ROUGE": llama_scores["rouge"][i],
        "Llama3.2_METEOR": llama_scores["meteor"][i],
        "Llama3.2_PPL": llama_scores["ppl"][i],
        "Llama3.2_BERT": llama_scores["bert"][i],
        "Llama3.2_BART": llama_scores["bart"][i],

        # Gemma
        "Gemma_BLEU": gemma_scores["bleu"][i],
        "Gemma_ROUGE": gemma_scores["rouge"][i],
        "Gemma_METEOR": gemma_scores["meteor"][i],
        "Gemma_PPL": gemma_scores["ppl"][i],
        "Gemma_BERT": gemma_scores["bert"][i],
        "Gemma_BART": gemma_scores["bart"][i],
    }
    final_rows.append(row)

print(f"Built {len(final_rows)} rows.")

df = pd.DataFrame(final_rows)
df.to_csv("model_comparison_results.csv", index=False)
print("Saved model_comparison_results.csv")


# ---------------------------------------------------------
# 5. Summary tables
# NOTE: model names here MUST match the column prefixes used above
# ("Gemma", not "Gemma2:2b" — a colon in the name would break the
# f-string column lookup below and raise a KeyError).
# ---------------------------------------------------------
metrics = ["BLEU", "ROUGE", "METEOR", "BERT", "BART"]
models = ["TinyLlama", "Llama3.2", "Gemma"]

summary = pd.DataFrame(
    {model: [df[f"{model}_{metric}"].mean() for metric in metrics] for model in models},
    index=metrics
)
print(summary)

summary_ppl = pd.DataFrame(
    {model: [df[f"{model}_PPL"].mean()] for model in models},
    index=["PPL"]
)
print(summary_ppl)


# ---------------------------------------------------------
# 6. Charts — saved to file as well as shown, so they survive outside the notebook
# ---------------------------------------------------------
summary.T.plot(kind="bar", figsize=(10, 6))
plt.title("Average Metric Scores by Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("model_comparison_metrics.png")
plt.show()

summary_ppl.T.plot(kind="bar", figsize=(10, 6))
plt.title("Average Perplexity by Model")
plt.ylabel("Perplexity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison_perplexity.png")
plt.show()
