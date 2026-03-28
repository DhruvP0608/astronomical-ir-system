import json

from models import (
    search_tfidf,
    search_tfidf_plus,
    search_bm25,
    search_dense
)

from ground_truth import queries


# =========================
# METRICS
# =========================

def precision_at_k(results, relevant, k=5):

    retrieved = [r[0] for r in results[:k]]

    rel = sum(1 for r in retrieved if r in relevant)

    return rel / k


def recall_at_k(results, relevant, k=5):

    retrieved = [r[0] for r in results[:k]]

    rel = sum(1 for r in retrieved if r in relevant)

    if len(relevant) == 0:
        return 0

    return rel / len(relevant)


def mrr(results, relevant):

    for i, r in enumerate(results):

        if r[0] in relevant:
            return 1 / (i + 1)

    return 0


def average_precision(results, relevant):

    score = 0
    rel_count = 0

    for i, r in enumerate(results):

        if r[0] in relevant:

            rel_count += 1

            score += rel_count / (i + 1)

    if rel_count == 0:
        return 0

    return score / rel_count


# =========================
# EVALUATION PIPELINE
# =========================

def evaluate(model_func):

    precisions = []
    recalls = []
    maps = []
    mrrs = []

    for q in queries:

        results = model_func(q["query"])

        relevant = q["relevant"]

        precisions.append(
            precision_at_k(results, relevant)
        )

        recalls.append(
            recall_at_k(results, relevant)
        )

        maps.append(
            average_precision(results, relevant)
        )

        mrrs.append(
            mrr(results, relevant)
        )

    return {

        "Precision@5": round(sum(precisions)/len(precisions),3),

        "Recall": round(sum(recalls)/len(recalls),3),

        "MAP": round(sum(maps)/len(maps),3),

        "MRR": round(sum(mrrs)/len(mrrs),3)

    }


print("\nBuilding models...")

print("Models ready.")


print("\nEvaluating TFIDF...")
tfidf = evaluate(search_tfidf)

print("Done.")


print("\nEvaluating TFIDF+...")
tfidf_plus = evaluate(search_tfidf_plus)

print("Done.")


print("\nEvaluating BM25...")
bm25 = evaluate(search_bm25)

print("Done.")


print("\nEvaluating Dense...")
dense = evaluate(search_dense)

print("Done.")


results = {

    "TFIDF": tfidf,

    "TFIDF+": tfidf_plus,

    "BM25": bm25,

    "Dense": dense

}


print("\nMODEL COMPARISON:\n")

for model, metrics in results.items():

    print(model)

    for metric, value in metrics.items():

        print(metric, value)

    print()


# =========================
# SAVE RESULTS
# =========================

with open("results.json","w") as f:

    json.dump(results, f, indent=4)


print("Results saved.")