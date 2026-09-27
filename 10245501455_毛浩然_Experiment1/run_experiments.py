"""Reproducible experiments for Course Experiment 1: text classification.

The official unlabeled test set is loaded for structural checks only until all
validation experiments and final model selection have finished.
"""

from __future__ import print_function

import argparse
import json
import os
import re
import shutil
import time
import warnings
from collections import Counter
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
SEEDS = [42, 2025, 3407]
C_VALUES = [0.1, 1.0, 10.0, 100.0]
RESULTS_DIR = "results"
BASE_TFIDF = {
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 30000,
    "lowercase": True,
    "strip_accents": "unicode",
}

HEADER_RE = re.compile(r"^[!-9;-~]+:\s*")
QUOTE_INTRO_RE = re.compile(r"^\s*(in article\b.*\bwrites:|on .+ wrote:)\s*$", re.I)


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def archive_existing_outputs(paths):
    existing = [path for path in paths if os.path.exists(path)]
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(RESULTS_DIR, "archive", stamp)
    os.makedirs(archive_dir)
    for path in existing:
        shutil.copy2(path, os.path.join(archive_dir, os.path.basename(path)))
    print("Existing outputs archived in:", archive_dir)
    return archive_dir


def remove_leading_header(text):
    """Remove the RFC-like header block at the beginning of a newsgroup post."""
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    end = 0
    saw_header = False
    for i, line in enumerate(lines):
        if not line.strip() and saw_header:
            end = i + 1
            break
        if HEADER_RE.match(line):
            saw_header = True
            continue
        if saw_header and (line.startswith(" ") or line.startswith("\t")):
            continue
        if saw_header:
            end = i
            break
        return "\n".join(lines)
    return "\n".join(lines[end:]) if saw_header else "\n".join(lines)


def clean_text(text, remove_header=False):
    """Apply small, explainable cleanup without stemming or external NLP tools."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if remove_header:
        text = remove_leading_header(text)
    kept = []
    for line in text.split("\n"):
        if line.lstrip().startswith(">") or QUOTE_INTRO_RE.match(line):
            continue
        kept.append(line.strip())
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def preprocess_texts(texts, mode):
    if mode == "raw_keep_header":
        return [str(x) for x in texts]
    if mode == "clean_keep_header":
        return [clean_text(x, remove_header=False) for x in texts]
    if mode == "clean_remove_header":
        return [clean_text(x, remove_header=True) for x in texts]
    raise ValueError("Unknown preprocessing mode: %s" % mode)


def make_model(model_name, seed=RANDOM_STATE, c_value=1.0):
    if model_name == "Multinomial Naive Bayes":
        return MultinomialNB(alpha=1.0)
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=c_value, max_iter=2000, solver="lbfgs", random_state=seed
        )
    if model_name == "Linear SVM":
        return LinearSVC(C=c_value, random_state=seed, max_iter=5000)
    if model_name == "MLPClassifier":
        return MLPClassifier(
            hidden_layer_sizes=(50,), alpha=0.0001, max_iter=100,
            early_stopping=True, n_iter_no_change=8, random_state=seed,
        )
    raise ValueError("Unknown model: %s" % model_name)


def model_parameters(model):
    params = model.get_params(deep=False)
    keep = ["alpha", "C", "solver", "max_iter", "random_state",
            "hidden_layer_sizes", "early_stopping", "activation"]
    return {key: params[key] for key in keep if key in params}


def evaluate_model(model, x_train, y_train, x_valid, y_valid):
    started = time.perf_counter()
    caught = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - started
        pred = model.predict(x_valid)
        for warning in records:
            if issubclass(warning.category, ConvergenceWarning):
                caught.append(str(warning.message))
    return {
        "accuracy": accuracy_score(y_valid, pred),
        "macro_f1": f1_score(y_valid, pred, average="macro"),
        "train_time": train_time,
        "prediction": pred,
        "warning": " | ".join(caught),
    }


def inspect_data(train_df, test_df, example_df):
    texts = train_df["text"].astype(str)
    class_counts = train_df["target"].value_counts().sort_index()
    lines = [
        "=== DATA INSPECTION ===",
        "train columns: %s" % list(train_df.columns),
        "train shape: %s" % (train_df.shape,),
        "test columns: %s" % list(test_df.columns),
        "test shape: %s" % (test_df.shape,),
        "prediction example shape (header=None): %s" % (example_df.shape,),
        "train missing values: %s" % train_df.isnull().sum().to_dict(),
        "test missing values: %s" % test_df.isnull().sum().to_dict(),
        "number of classes: %d" % train_df["target"].nunique(),
        "class counts:",
        class_counts.to_string(),
    ]
    for field in ["Subject:", "From:", "Organization:"]:
        count = int(texts.str.contains(re.escape(field), case=False, regex=True).sum())
        lines.append("%s present: %d/%d (%.2f%%)" %
                     (field, count, len(texts), 100.0 * count / len(texts)))
    lengths = texts.str.len()
    lines.append("text length chars min/median/mean/max: %d/%.1f/%.2f/%d" %
                 (lengths.min(), lengths.median(), lengths.mean(), lengths.max()))
    lines.append("\n=== FIVE TRAINING TEXT EXCERPTS (first 500 characters) ===")
    for idx in [0, 1, 2, 10, 100]:
        excerpt = texts.iloc[idx][:500].replace("\x00", " ")
        lines.extend(["--- index %d target %s ---" % (idx, train_df.iloc[idx]["target"]), excerpt])
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS_DIR, "data_inspection.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return class_counts


def experiment_a(train_texts, valid_texts, y_train, y_valid):
    print("\n=== EXPERIMENT A: BASELINE MODELS ===")
    vectorizer = TfidfVectorizer(**BASE_TFIDF)
    started = time.perf_counter()
    x_train = vectorizer.fit_transform(train_texts)
    x_valid = vectorizer.transform(valid_texts)
    vectorize_time = time.perf_counter() - started
    print("Shared TF-IDF shape:", x_train.shape,
          "vectorization time: %.3fs" % vectorize_time)
    rows = []
    for name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM", "MLPClassifier"]:
        model = make_model(name)
        metrics = evaluate_model(model, x_train, y_train, x_valid, y_valid)
        row = {
            "model": name,
            "parameters": json_text(model_parameters(model)),
            "tfidf_parameters": json_text(BASE_TFIDF),
            "n_features": x_train.shape[1],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "train_time": metrics["train_time"],
            "vectorization_time": vectorize_time,
            "warning": metrics["warning"],
        }
        rows.append(row)
        print("%-25s accuracy=%.4f macro_f1=%.4f train_time=%.3fs%s" %
              (name, row["accuracy"], row["macro_f1"], row["train_time"],
               " WARNING=" + row["warning"] if row["warning"] else ""))
    result = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    result.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False,
                  encoding="utf-8-sig")
    return result, vectorizer


def experiment_b(baseline, train_texts, valid_texts, y_train, y_valid):
    print("\n=== EXPERIMENT B: C TUNING ===")
    supported = baseline[baseline["model"].isin(["Logistic Regression", "Linear SVM"])]
    top_models = supported.sort_values(["macro_f1", "accuracy"], ascending=False)["model"].head(2).tolist()
    print("Selected models based on baseline validation Macro-F1:", top_models)
    vectorizer = TfidfVectorizer(**BASE_TFIDF)
    x_train = vectorizer.fit_transform(train_texts)
    x_valid = vectorizer.transform(valid_texts)
    individual = []
    for name in top_models:
        for c_value in C_VALUES:
            for seed in SEEDS:
                model = make_model(name, seed=seed, c_value=c_value)
                metrics = evaluate_model(model, x_train, y_train, x_valid, y_valid)
                individual.append({
                    "record_type": "run", "model": name, "C": c_value,
                    "random_state": seed, "parameters": json_text(model_parameters(model)),
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "train_time": metrics["train_time"], "accuracy_mean": np.nan,
                    "accuracy_std": np.nan, "macro_f1_mean": np.nan,
                    "macro_f1_std": np.nan, "train_time_mean": np.nan,
                    "train_time_std": np.nan, "warning": metrics["warning"],
                })
                print("%-19s C=%-5g seed=%-4d acc=%.4f f1=%.4f time=%.3fs" %
                      (name, c_value, seed, metrics["accuracy"],
                       metrics["macro_f1"], metrics["train_time"]))
    runs = pd.DataFrame(individual)
    summaries = []
    for (name, c_value), group in runs.groupby(["model", "C"], sort=False):
        summaries.append({
            "record_type": "summary", "model": name, "C": c_value,
            "random_state": "all", "parameters": json_text({"C": c_value, "seeds": SEEDS}),
            "accuracy": np.nan, "macro_f1": np.nan, "train_time": np.nan,
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(ddof=0),
            "macro_f1_mean": group["macro_f1"].mean(),
            "macro_f1_std": group["macro_f1"].std(ddof=0),
            "train_time_mean": group["train_time"].mean(),
            "train_time_std": group["train_time"].std(ddof=0),
            "warning": " | ".join(x for x in group["warning"].unique() if x),
        })
    summary = pd.DataFrame(summaries)
    output = pd.concat([runs, summary], ignore_index=True, sort=False)
    output.to_csv(os.path.join(RESULTS_DIR, "tuning_results.csv"), index=False,
                  encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, group in summary.groupby("model"):
        group = group.sort_values("C")
        ax.plot(group["C"], group["macro_f1_mean"], marker="o", linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization parameter C (log scale)")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Effect of C on validation performance")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "parameter_curve.png"), dpi=180)
    plt.close(fig)
    best = summary.sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False).iloc[0]
    print("Best tuning configuration: %s, C=%g, mean Macro-F1=%.4f" %
          (best["model"], best["C"], best["macro_f1_mean"]))
    return output, summary, str(best["model"]), float(best["C"])


def experiment_c(model_name, c_value, train_texts, valid_texts, y_train, y_valid):
    print("\n=== EXPERIMENT C: TF-IDF / PREPROCESSING ABLATION ===")
    configurations = [
        ("raw_keep_header", (1, 1), 80000, "A Raw text + unigram"),
        ("raw_keep_header", (1, 2), 80000, "B Raw text + unigram/bigram"),
        ("clean_keep_header", (1, 1), 80000, "C Clean text + unigram"),
        ("clean_keep_header", (1, 2), 80000, "D Clean text + unigram/bigram"),
        ("clean_remove_header", (1, 1), 80000, "E Clean text, headers removed + unigram"),
        ("clean_remove_header", (1, 2), 80000, "F Clean text, headers removed + unigram/bigram"),
        ("raw_keep_header", (1, 1), 30000, "G Raw unigram + 30k feature cap control"),
    ]
    cache = {}
    rows = []
    for mode, ngram, max_features, label in configurations:
        if mode not in cache:
            cache[mode] = (preprocess_texts(train_texts, mode),
                           preprocess_texts(valid_texts, mode))
        current_train, current_valid = cache[mode]
        tfidf_params = dict(BASE_TFIDF)
        tfidf_params.update({"ngram_range": ngram, "max_features": max_features})
        vectorizer = TfidfVectorizer(**tfidf_params)
        started = time.perf_counter()
        x_train = vectorizer.fit_transform(current_train)
        x_valid = vectorizer.transform(current_valid)
        vectorize_time = time.perf_counter() - started
        model = make_model(model_name, c_value=c_value)
        metrics = evaluate_model(model, x_train, y_train, x_valid, y_valid)
        row = {
            "experiment": label,
            "preprocessing": mode,
            "ngram_range": str(ngram),
            "tfidf_parameters": json_text(tfidf_params),
            "model": model_name,
            "model_parameters": json_text(model_parameters(model)),
            "n_features": x_train.shape[1],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "train_time": vectorize_time + metrics["train_time"],
            "vectorization_time": vectorize_time,
            "model_train_time": metrics["train_time"],
            "warning": metrics["warning"],
        }
        rows.append(row)
        print("%-45s features=%-6d acc=%.4f f1=%.4f total_time=%.3fs" %
              (label, row["n_features"], row["accuracy"], row["macro_f1"], row["train_time"]))
    result = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    result.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False,
                  encoding="utf-8-sig")
    return result


def create_error_analysis(best_row, train_texts, valid_texts, y_train, y_valid):
    mode = best_row["preprocessing"]
    processed_train = preprocess_texts(train_texts, mode)
    processed_valid = preprocess_texts(valid_texts, mode)
    params = json.loads(best_row["tfidf_parameters"])
    params["ngram_range"] = tuple(params["ngram_range"])
    ngram = params["ngram_range"]
    vectorizer = TfidfVectorizer(**params)
    x_train = vectorizer.fit_transform(processed_train)
    x_valid = vectorizer.transform(processed_valid)
    model_params = json.loads(best_row["model_parameters"])
    c_value = float(model_params.get("C", 1.0))
    model = make_model(best_row["model"], c_value=c_value)
    metrics = evaluate_model(model, x_train, y_train, x_valid, y_valid)
    pred = metrics["prediction"]
    labels = sorted(np.unique(y_valid).tolist())
    cm = confusion_matrix(y_valid, pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label",
           title="Validation confusion matrix")
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=180)
    plt.close(fig)

    report = classification_report(y_valid, pred, labels=labels, digits=4)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("Final validation configuration\n")
        fh.write("model=%s\npreprocessing=%s\nngram_range=%s\nC=%g\n\n" %
                 (best_row["model"], mode, ngram, c_value))
        fh.write(report)

    off_diagonal = Counter()
    for true, guessed in zip(y_valid, pred):
        if true != guessed:
            off_diagonal[(int(true), int(guessed))] += 1
    top_pairs = off_diagonal.most_common(5)
    examples = []
    pair_used = Counter()
    for index, (text, true, guessed) in enumerate(zip(valid_texts, y_valid, pred)):
        pair = (int(true), int(guessed))
        if pair in dict(top_pairs) and pair_used[pair] < 3:
            examples.append({
                "validation_position": index,
                "true_label": pair[0],
                "predicted_label": pair[1],
                "pair_error_count": off_diagonal[pair],
                "text_excerpt": re.sub(r"\s+", " ", str(text))[:700],
            })
            pair_used[pair] += 1
    pd.DataFrame(examples).to_csv(os.path.join(RESULTS_DIR, "misclassified_examples.csv"),
                                  index=False, encoding="utf-8-sig")
    return metrics, cm, top_pairs, report


def format_metric(value):
    return "%.4f" % float(value)


def write_summary(train_df, test_df, y_train, y_valid, baseline, tuning_summary,
                  ablation, best_row, final_metrics, top_pairs, original_example_shape):
    best_two = baseline.sort_values(["macro_f1", "accuracy"], ascending=False).head(2)
    baseline_table = baseline[["model", "accuracy", "macro_f1", "train_time"]].copy()
    for col in ["accuracy", "macro_f1", "train_time"]:
        baseline_table[col] = baseline_table[col].map(lambda x: "%.4f" % x)
    tuning_table = tuning_summary[["model", "C", "accuracy_mean", "accuracy_std",
                                   "macro_f1_mean", "macro_f1_std", "train_time_mean"]].copy()
    for col in ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "train_time_mean"]:
        tuning_table[col] = tuning_table[col].map(lambda x: "%.4f" % x)
    ablation_table = ablation[["experiment", "n_features", "accuracy", "macro_f1", "train_time"]].copy()
    for col in ["accuracy", "macro_f1", "train_time"]:
        ablation_table[col] = ablation_table[col].map(lambda x: "%.4f" % x)

    raw_uni = ablation[ablation["experiment"] == "A Raw text + unigram"].iloc[0]
    raw_bi = ablation[ablation["experiment"] == "B Raw text + unigram/bigram"].iloc[0]
    clean_uni = ablation[ablation["experiment"] == "C Clean text + unigram"].iloc[0]
    clean_bi = ablation[ablation["experiment"] == "D Clean text + unigram/bigram"].iloc[0]
    nohead_bi = ablation[ablation["experiment"] == "F Clean text, headers removed + unigram/bigram"].iloc[0]
    class_counts = train_df["target"].value_counts().sort_index().to_dict()
    warnings_seen = []
    for frame in [baseline, ablation]:
        warnings_seen.extend([str(x) for x in frame["warning"].fillna("") if str(x)])
    warnings_text = "No convergence warnings were observed." if not warnings_seen else "Warnings: " + " | ".join(sorted(set(warnings_seen)))

    findings = [
        "Increasing from unigram to unigram+bigram on raw text changed Macro-F1 from %s to %s (%+.4f), while feature count changed from %d to %d." %
        (format_metric(raw_uni["macro_f1"]), format_metric(raw_bi["macro_f1"]),
         raw_bi["macro_f1"] - raw_uni["macro_f1"], raw_uni["n_features"], raw_bi["n_features"]),
        "With unigram features, light cleaning changed Macro-F1 from %s to %s (%+.4f)." %
        (format_metric(raw_uni["macro_f1"]), format_metric(clean_uni["macro_f1"]),
         clean_uni["macro_f1"] - raw_uni["macro_f1"]),
        "With unigram+bigram features and cleaning, removing the leading header block changed Macro-F1 from %s to %s (%+.4f)." %
        (format_metric(clean_bi["macro_f1"]), format_metric(nohead_bi["macro_f1"]),
         nohead_bi["macro_f1"] - clean_bi["macro_f1"]),
    ]
    svm_curve = tuning_summary[tuning_summary["model"] == "Linear SVM"].set_index("C")
    findings.append(
        "For Linear SVM, Macro-F1 increased from %.4f at C=0.1 to %.4f at C=1, then decreased to %.4f at C=10 and %.4f at C=100." %
        (svm_curve.loc[0.1, "macro_f1_mean"], svm_curve.loc[1.0, "macro_f1_mean"],
         svm_curve.loc[10.0, "macro_f1_mean"], svm_curve.loc[100.0, "macro_f1_mean"]))
    content = """# Experiment record summary

## 1. Data and split

- Official training data: {n_train} rows, columns `text` and `target`; no missing values.
- Official unlabeled test data: {n_test} rows, one `text` column; no missing values.
- Classes: {n_classes}; counts: `{class_counts}`.
- Header prevalence: `Subject` 100%, `From` 100%, `Organization` 96.04% in training text.
- Split: `train_test_split(test_size=0.20, stratify=y, random_state=42)`, giving {split_train} training and {split_valid} validation rows.
- The official test set was not transformed or used for model selection. It was used only after the validation configuration was fixed.
- Original prediction example: {example_rows} rows, one integer column, no header.

## 2. Baseline models

{baseline_table}

The top two validation models were **{best1}** and **{best2}**, selected by Macro-F1, with Accuracy as the tie-breaker.
MLP was therefore the second-best baseline, but it has no `C` parameter and took much longer to fit. Following the requested C study, Experiment B compares the two C-based linear models, Linear SVM and Logistic Regression.

## 3. C tuning

{tuning_table}

All C experiments used the same fixed split. The three values 42, 2025 and 3407 were model seeds, not new splits. Standard deviations are reported even when zero; zero indicates that the fitted estimator was effectively deterministic under this setup.

## 4. TF-IDF and preprocessing ablation

{ablation_table}

Cleaning removes quoted lines beginning with `>`, common quote-introduction lines, and redundant whitespace. Header removal separately deletes the complete leading RFC-like header block. No stemming, lemmatization, or external NLP model was used.

## 5. Final validation configuration

- Model: **{model}**
- Model parameters: `{model_parameters}`
- Preprocessing: **{preprocessing}**
- N-grams: **{ngram}**
- TF-IDF parameters: `{tfidf_parameters}`
- Validation Accuracy: **{accuracy}**
- Validation Macro-F1: **{macro_f1}**
- Most frequent directed confusions (true -> predicted, count): `{top_pairs}`

## 6. Findings supported by the runs

1. {finding1}
2. {finding2}
3. {finding3}
4. {finding4}

## 7. Warnings and limitations

- {warnings_text}
- Training time is wall-clock time on this computer and can vary with load. Baseline `train_time` is estimator fit time because all models share one precomputed matrix; ablation `train_time` includes vectorization plus estimator fitting.
- Only one stratified hold-out split was used, as required. This is adequate for the controlled course experiment but less robust than cross-validation.
- Labels are numeric because no category-name mapping is supplied in the CSV files.
- The bigram settings cap features at 80,000 to bound memory use; the actual retained feature count is recorded for every ablation.
- A supplementary 30,000-feature unigram control reproduces the tuned baseline representation so final selection considers the best validation result from all stages.
- The installed scikit-learn 1.8 rejects direct multiclass fitting with `liblinear`, so Logistic Regression uses the standard multiclass-capable `lbfgs` solver.
""".format(
        n_train=len(train_df), n_test=len(test_df), n_classes=train_df["target"].nunique(),
        class_counts=class_counts, split_train=len(y_train), split_valid=len(y_valid),
        example_rows=original_example_shape[0], baseline_table=baseline_table.to_string(index=False),
        best1=best_two.iloc[0]["model"], best2=best_two.iloc[1]["model"],
        tuning_table=tuning_table.to_string(index=False),
        ablation_table=ablation_table.to_string(index=False), model=best_row["model"],
        model_parameters=best_row["model_parameters"], preprocessing=best_row["preprocessing"],
        ngram=best_row["ngram_range"], tfidf_parameters=best_row["tfidf_parameters"],
        accuracy=format_metric(final_metrics["accuracy"]), macro_f1=format_metric(final_metrics["macro_f1"]),
        top_pairs=top_pairs, finding1=findings[0], finding2=findings[1], finding3=findings[2],
        finding4=findings[3],
        warnings_text=warnings_text)
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(content)


def train_final_and_predict(train_df, test_df, best_row):
    print("\n=== FINAL TRAINING AND TEST PREDICTION ===")
    mode = best_row["preprocessing"]
    train_text = preprocess_texts(train_df["text"].astype(str).tolist(), mode)
    test_text = preprocess_texts(test_df["text"].astype(str).tolist(), mode)
    tfidf_params = json.loads(best_row["tfidf_parameters"])
    tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
    vectorizer = TfidfVectorizer(**tfidf_params)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    model_params = json.loads(best_row["model_parameters"])
    model = make_model(best_row["model"], c_value=float(model_params.get("C", 1.0)))
    model.fit(x_train, train_df["target"].values)
    prediction = model.predict(x_test)
    pd.DataFrame(prediction).to_csv("predictions.csv", index=False, header=False)
    check = pd.read_csv("predictions.csv", header=None)
    if check.shape != (len(test_df), 1):
        raise RuntimeError("Prediction shape mismatch: %s" % (check.shape,))
    if check[0].isnull().any():
        raise RuntimeError("Predictions contain missing values")
    if not set(check[0].unique()).issubset(set(train_df["target"].unique())):
        raise RuntimeError("Predictions contain unknown labels")
    print("Final feature matrix:", x_train.shape)
    print("predictions.csv verified: %d rows, 1 column, header=False, labels=%s" %
          (len(check), sorted(check[0].unique().tolist())))


def main():
    parser = argparse.ArgumentParser(description="Run all controlled text classification experiments.")
    parser.add_argument("--skip-archive", action="store_true",
                        help="Do not archive prior generated outputs before rerunning.")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    generated = [
        os.path.join(RESULTS_DIR, name) for name in [
            "data_inspection.txt", "baseline_results.csv", "tuning_results.csv",
            "ablation_results.csv", "parameter_curve.png", "confusion_matrix.png",
            "classification_report.txt", "misclassified_examples.csv", "summary.md",
        ]
    ]
    if not args.skip_archive:
        archive_existing_outputs(generated)

    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")
    example_snapshot = os.path.join(RESULTS_DIR, "predictions_format_example_original.csv")
    if not os.path.exists(example_snapshot):
        shutil.copy2("predictions.csv", example_snapshot)
    example_df = pd.read_csv(example_snapshot, header=None)
    required_train = {"text", "target"}
    if set(train_df.columns) != required_train or list(test_df.columns) != ["text"]:
        raise ValueError("Unexpected CSV columns")
    if train_df.isnull().any().any() or test_df.isnull().any().any():
        raise ValueError("Missing values found; inspect before proceeding")
    inspect_data(train_df, test_df, example_df)

    indices = np.arange(len(train_df))
    train_idx, valid_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE,
        stratify=train_df["target"].values,
    )
    texts = train_df["text"].astype(str).tolist()
    labels = train_df["target"].values
    train_texts = [texts[i] for i in train_idx]
    valid_texts = [texts[i] for i in valid_idx]
    y_train, y_valid = labels[train_idx], labels[valid_idx]
    print("\nFixed split: training=%d, validation=%d, random_state=%d, stratified=True" %
          (len(y_train), len(y_valid), RANDOM_STATE))

    baseline, _ = experiment_a(train_texts, valid_texts, y_train, y_valid)
    tuning, tuning_summary, tuned_model, tuned_c = experiment_b(
        baseline, train_texts, valid_texts, y_train, y_valid)
    ablation = experiment_c(tuned_model, tuned_c, train_texts, valid_texts, y_train, y_valid)
    best_row = ablation.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    print("\nSelected final validation configuration:")
    print(best_row[["experiment", "model", "model_parameters", "accuracy", "macro_f1"]].to_string())
    final_metrics, cm, top_pairs, report = create_error_analysis(
        best_row, train_texts, valid_texts, y_train, y_valid)
    write_summary(train_df, test_df, y_train, y_valid, baseline, tuning_summary,
                  ablation, best_row, final_metrics, top_pairs, example_df.shape)
    train_final_and_predict(train_df, test_df, best_row)
    print("\nAll requested outputs were written successfully.")


if __name__ == "__main__":
    main()
