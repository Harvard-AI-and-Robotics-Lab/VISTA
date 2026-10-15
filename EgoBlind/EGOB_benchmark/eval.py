# Example:
# python eval.py --test_path test.csv --outputs_dir outputs

# Imports
import argparse
import json
from pathlib import Path

import pandas as pd
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


# setup
ANSWER_COLUMNS = ["answer0", "answer1", "answer2", "answer3"]
METRIC_COLUMNS = ["bleu", "rouge1", "rouge2", "rougeL", "sbert_score", "bert_score"]
SUMMARY_COLUMNS = ["split", *METRIC_COLUMNS]


# args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute BLEU, ROUGE, SBERT, and BERTScore summaries for benchmark outputs."
    )
    parser.add_argument("--test_path", default="test.csv", help="Path to test.csv.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing model output folders.")
    parser.add_argument("--pred_path", default=None, help="Optional single predictions_*.jsonl file.")
    parser.add_argument("--sbert_model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--bert_model", default="roberta-large")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def read_predictions(prediction_path):
    prediction_rows = []
    with open(prediction_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {prediction_path} line {line_number}") from exc
            if "question_id" not in item or "pred" not in item:
                continue
            prediction_rows.append({
                "question_id": item["question_id"],
                "pred": "" if item["pred"] is None else str(item["pred"]).strip(),
            })
    return pd.DataFrame(prediction_rows)


def read_test_data(test_path):
    test_data = pd.read_csv(test_path)
    test_data[["type", *ANSWER_COLUMNS]] = test_data[["type", *ANSWER_COLUMNS]].fillna("").astype(str)
    return test_data[["question_id", "type", *ANSWER_COLUMNS]]


def get_reference_answers(row):
    references = [row.get(column, "").strip() for column in ANSWER_COLUMNS]
    return [reference for reference in references if reference]


# BLEU and ROUGE
def best_rouge_scores(prediction, references, scorer):
    all_scores = [scorer.score(reference, prediction) for reference in references]
    return {
        metric_name: max(scores[metric_name].fmeasure for scores in all_scores)
        for metric_name in ["rouge1", "rouge2", "rougeL"]
    }


def add_bleu_rouge_scores(results):
    # BLEU and ROUGE compare prediction against the reference answers
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bleu_scores = []
    rouge_scores = []

    for _, row in results.iterrows():
        prediction = row["pred"]
        references = row["references"]
        if not prediction or not references:
            bleu_scores.append(0.0)
            rouge_scores.append({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
            continue

        bleu_scores.append(sacrebleu.sentence_bleu(prediction, references).score / 100.0)
        rouge_scores.append(best_rouge_scores(prediction, references, scorer))

    results["bleu"] = bleu_scores
    results["rouge1"] = [item["rouge1"] for item in rouge_scores]
    results["rouge2"] = [item["rouge2"] for item in rouge_scores]
    results["rougeL"] = [item["rougeL"] for item in rouge_scores]
    return results


# SBERT
def add_sentence_similarity_scores(results, model, batch_size):
    predictions = results["pred"].tolist()
    prediction_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarity_scores = []
    for index, references in enumerate(results["references"].tolist()):
        if not predictions[index] or not references:
            similarity_scores.append(0.0)
            continue
        reference_embeddings = model.encode(references, batch_size=batch_size, 
                normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = util.cos_sim(reference_embeddings, prediction_embeddings[index])
        similarity_scores.append(float(similarities.max()))

    results["sbert_score"] = similarity_scores
    return results


# BERTScore
def add_bert_similarity_scores(results, model_name, batch_size):
    candidate_answers = []
    reference_answers = []
    question_ids = []

    for _, row in results.iterrows():
        if not row["pred"] or not row["references"]:
            continue
        for reference in row["references"]:
            candidate_answers.append(row["pred"])
            reference_answers.append(reference)
            question_ids.append(row["question_id"])

    scores_by_question_id = {question_id: 0.0 for question_id in results["question_id"].tolist()}
    if candidate_answers:
        _, _, f1_scores = bert_score(
            candidate_answers,
            reference_answers,
            model_type=model_name,
            lang="en",
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=False,
        )
        for question_id, score in zip(question_ids, f1_scores.tolist()):
            scores_by_question_id[question_id] = max(scores_by_question_id[question_id], float(score))

    results["bert_score"] = results["question_id"].map(scores_by_question_id).fillna(0.0)
    return results


def create_summary(results):
    summary_rows = []

    def make_summary_row(label, subset):
        return {
            "split": label,
            **subset[METRIC_COLUMNS].mean().to_dict(),
        }

    summary_rows.append(make_summary_row("overall", results))
    for type_name in sorted(results["type"].dropna().unique()):
        type_results = results[results["type"] == type_name]
        summary_rows.append(make_summary_row(type_name, type_results))

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary[METRIC_COLUMNS] = summary[METRIC_COLUMNS].round(4)
    return summary


def write_summary(summary, prediction_path):
    output_path = prediction_path.parent / f"summary_{prediction_path.parent.name}.csv"
    summary.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {output_path}")
    return output_path


def prepare_results(predictions, test_data):
    results = predictions.merge(test_data, on="question_id", how="inner")
    results["references"] = results.apply(get_reference_answers, axis=1)
    results["type"] = results["type"].str.strip().str.lower().str.rstrip(".")
    return results


def evaluate_prediction_file(prediction_path, test_data, sbert_model, args):
    predictions = read_predictions(prediction_path)
    if predictions.empty:
        print(f"[WARNING] No predictions found in {prediction_path}")
        return None

    results = prepare_results(predictions, test_data)
    missing_count = len(predictions) - len(results)
    if missing_count:
        print(f"[WARNING] {prediction_path}: skipped {missing_count} predictions without test.csv rows")

    results = add_bleu_rouge_scores(results)
    results = add_sentence_similarity_scores(results, sbert_model, args.batch_size)
    results = add_bert_similarity_scores(results, args.bert_model, args.batch_size)

    summary = create_summary(results)
    return write_summary(summary, prediction_path)


def find_prediction_files(args):
    if args.pred_path:
        return [Path(args.pred_path)]
    outputs_dir = Path(args.outputs_dir)
    return sorted(outputs_dir.glob("*/predictions_*.jsonl"))


def main():
    args = parse_args()
    test_data = read_test_data(args.test_path)

    # Load SBERT model
    sbert_model = SentenceTransformer(args.sbert_model)

    prediction_files = find_prediction_files(args)
    if not prediction_files:
        raise FileNotFoundError(f"No predictions_*.jsonl files found under {args.outputs_dir}")

    print(f"[INFO] Found {len(prediction_files)} prediction file(s)")
    for prediction_path in prediction_files:
        print(f"[INFO] Evaluating {prediction_path}")
        evaluate_prediction_file(prediction_path, test_data, sbert_model, args)


if __name__ == "__main__":
    main()
