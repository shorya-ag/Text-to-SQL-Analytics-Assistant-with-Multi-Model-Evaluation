import pickle
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load all three pickle files
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

# 2. Sanity checks - same question count and same question order across all three
results = tinyllama_data["results"]  # use TinyLlama's as the reference question order
expected_len = len(results)

for model_name, data in [("TinyLlama", tinyllama_data), ("Llama", llama_data), ("Gemma", gemma_data)]:
    for metric in ["bleu", "rouge", "meteor", "ppl", "bert", "bart"]:
        if len(data[metric]) != expected_len:
            print(f"MISMATCH: {model_name} - {metric} has {len(data[metric])}, expected {expected_len}")

for i in range(expected_len):
    q_tiny = tinyllama_data["results"][i]["user_input"]
    q_llama = llama_data["results"][i]["user_input"]
    q_gemma = gemma_data["results"][i]["user_input"]
    if not (q_tiny == q_llama == q_gemma):
        print(f"Question mismatch at index {i}: {q_tiny} | {q_llama} | {q_gemma}")

print("Sanity check complete.")

# 3. Pull out per-model score lists
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

# 4. Labeling logic - Good / Average / Bad per metric
def label_fixed(value, good_threshold, avg_threshold, higher_is_better=True):
    if higher_is_better:
        if value >= good_threshold:
            return "Good"
        elif value >= avg_threshold:
            return "Average"
        return "Bad"
    else:
        if value <= good_threshold:
            return "Good"
        elif value <= avg_threshold:
            return "Average"
        return "Bad"


def label_bleu(value):
    return label_fixed(value, good_threshold=0.5, avg_threshold=0.2, higher_is_better=True)


def label_rouge(value):
    return label_fixed(value, good_threshold=0.5, avg_threshold=0.2, higher_is_better=True)


def label_meteor(value):
    return label_fixed(value, good_threshold=0.5, avg_threshold=0.2, higher_is_better=True)


def label_bert(value):
    return label_fixed(value, good_threshold=0.6, avg_threshold=0.3, higher_is_better=True)


def label_ppl(value):
    return label_fixed(value, good_threshold=30, avg_threshold=60, higher_is_better=False)


def label_bart_relative(value, all_bart_values):
    """BART logits have no fixed scale, so label relative to the pooled
    distribution across all models/questions using tertiles."""
    sorted_vals = sorted(all_bart_values)
    n = len(sorted_vals)
    lower_cut = sorted_vals[n // 3]
    upper_cut = sorted_vals[(2 * n) // 3]
    if value >= upper_cut:
        return "Good"
    elif value >= lower_cut:
        return "Average"
    return "Bad"


# Pool all BART scores across all three models to compute tertile cutoffs once
all_bart_values = tinyllama_scores["bart"] + llama_scores["bart"] + gemma_scores["bart"]

# 5. Build one row per question (1-based index), with all three
#    models' scores + labels side by side, and a Ground Truth column

final_rows = []

for i, item in enumerate(results, start=1):
    row = {
        "Index": i,
        "Query": item["user_input"],
        "Reference": item["reference"],
        "Ground Truth": item["reference"],  # same source as Reference unless tracked separately

        # TinyLlama
        "TinyLlama_BLEU": tinyllama_scores["bleu"][i - 1],
        "TinyLlama_BLEU_Label": label_bleu(tinyllama_scores["bleu"][i - 1]),
        "TinyLlama_ROUGE": tinyllama_scores["rouge"][i - 1],
        "TinyLlama_ROUGE_Label": label_rouge(tinyllama_scores["rouge"][i - 1]),
        "TinyLlama_METEOR": tinyllama_scores["meteor"][i - 1],
        "TinyLlama_METEOR_Label": label_meteor(tinyllama_scores["meteor"][i - 1]),
        "TinyLlama_PPL": tinyllama_scores["ppl"][i - 1],
        "TinyLlama_PPL_Label": label_ppl(tinyllama_scores["ppl"][i - 1]),
        "TinyLlama_BERT": tinyllama_scores["bert"][i - 1],
        "TinyLlama_BERT_Label": label_bert(tinyllama_scores["bert"][i - 1]),
        "TinyLlama_BART": tinyllama_scores["bart"][i - 1],
        "TinyLlama_BART_Label": label_bart_relative(tinyllama_scores["bart"][i - 1], all_bart_values),

        # Llama 3.2
        "Llama_BLEU": llama_scores["bleu"][i - 1],
        "Llama_BLEU_Label": label_bleu(llama_scores["bleu"][i - 1]),
        "Llama_ROUGE": llama_scores["rouge"][i - 1],
        "Llama_ROUGE_Label": label_rouge(llama_scores["rouge"][i - 1]),
        "Llama_METEOR": llama_scores["meteor"][i - 1],
        "Llama_METEOR_Label": label_meteor(llama_scores["meteor"][i - 1]),
        "Llama_PPL": llama_scores["ppl"][i - 1],
        "Llama_PPL_Label": label_ppl(llama_scores["ppl"][i - 1]),
        "Llama_BERT": llama_scores["bert"][i - 1],
        "Llama_BERT_Label": label_bert(llama_scores["bert"][i - 1]),
        "Llama_BART": llama_scores["bart"][i - 1],
        "Llama_BART_Label": label_bart_relative(llama_scores["bart"][i - 1], all_bart_values),

        # Gemma
        "Gemma_BLEU": gemma_scores["bleu"][i - 1],
        "Gemma_BLEU_Label": label_bleu(gemma_scores["bleu"][i - 1]),
        "Gemma_ROUGE": gemma_scores["rouge"][i - 1],
        "Gemma_ROUGE_Label": label_rouge(gemma_scores["rouge"][i - 1]),
        "Gemma_METEOR": gemma_scores["meteor"][i - 1],
        "Gemma_METEOR_Label": label_meteor(gemma_scores["meteor"][i - 1]),
        "Gemma_PPL": gemma_scores["ppl"][i - 1],
        "Gemma_PPL_Label": label_ppl(gemma_scores["ppl"][i - 1]),
        "Gemma_BERT": gemma_scores["bert"][i - 1],
        "Gemma_BERT_Label": label_bert(gemma_scores["bert"][i - 1]),
        "Gemma_BART": gemma_scores["bart"][i - 1],
        "Gemma_BART_Label": label_bart_relative(gemma_scores["bart"][i - 1], all_bart_values),
    }
    final_rows.append(row)

print(f"Built {len(final_rows)} rows.")

df = pd.DataFrame(final_rows)
df.to_csv("model_comparison_results.csv", index=False)
print("Saved model_comparison_results.csv")

# 6. Summary tables (raw scores only - label columns are per-question detail,
#    not meaningfully averaged)

metrics = ["BLEU", "ROUGE", "METEOR", "BERT", "BART"]
models = ["TinyLlama", "Llama", "Gemma"]

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

# 7. Charts
summary.T.plot(kind="bar", figsize=(10, 6))
plt.title("Average Metric Scores by Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

summary_ppl.T.plot(kind="bar", figsize=(10, 6))
plt.title("Average Perplexity by Model")
plt.ylabel("Perplexity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()