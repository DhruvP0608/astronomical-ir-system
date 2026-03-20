import numpy as np

from models import TFIDFModel
from models import BM25Model
from models import DenseModel
from models import CorpusLoader

from ground_truth import evaluation_queries



# =========================
# METRIC FUNCTIONS
# =========================

def precision_at_k(results, relevant, k):

    retrieved = [doc for doc,score in results[:k]]

    relevant_found = len(
        set(retrieved).intersection(set(relevant))
    )

    return relevant_found / k



def recall_at_k(results, relevant, k):

    retrieved = [doc for doc,score in results[:k]]

    relevant_found = len(
        set(retrieved).intersection(set(relevant))
    )

    return relevant_found / len(relevant)



def average_precision(results, relevant):

    hits = 0
    score = 0

    for i,(doc,_) in enumerate(results):

        if doc in relevant:

            hits += 1

            score += hits/(i+1)

    if hits == 0:
        return 0

    return score / len(relevant)



def reciprocal_rank(results, relevant):

    for i,(doc,_) in enumerate(results):

        if doc in relevant:
            return 1/(i+1)

    return 0



# =========================
# EVALUATION ENGINE
# =========================

class ModelEvaluator:

    def __init__(self):

        corpus = CorpusLoader("../astronomical_corpus.csv")

        self.documents = corpus.get_documents()

        self.doc_ids = corpus.get_doc_ids()

        self.tokenized = corpus.get_tokenized_documents()


        self.tfidf = TFIDFModel(
            self.documents,
            self.doc_ids
        )

        self.bm25 = BM25Model(
            self.tokenized,
            self.doc_ids
        )

        self.dense = DenseModel(
            self.documents,
            self.doc_ids
        )


        print("Building models...")

        self.tfidf.build()

        self.bm25.build()

        self.dense.build()

        print("Models ready.")


    def evaluate_model(self, model):

        p5 = []
        r5 = []
        ap = []
        rr = []

        for query in evaluation_queries:

            relevant = evaluation_queries[query]

            results = model.search(query,10)

            p5.append(
                precision_at_k(results,relevant,5)
            )

            r5.append(
                recall_at_k(results,relevant,5)
            )

            ap.append(
                average_precision(results,relevant)
            )

            rr.append(
                reciprocal_rank(results,relevant)
            )

        return {

            "Precision@5": np.mean(p5),

            "Recall@5": np.mean(r5),

            "MAP": np.mean(ap),

            "MRR": np.mean(rr)

        }


    def run(self):

        results = {}

        print("\nEvaluating TFIDF...")

        results["TFIDF"] = self.evaluate_model(
            self.tfidf
        )

        print("Done.")


        print("\nEvaluating BM25...")

        results["BM25"] = self.evaluate_model(
            self.bm25
        )

        print("Done.")


        print("\nEvaluating Dense...")

        results["Dense"] = self.evaluate_model(
            self.dense
        )

        print("Done.")


        return results



# =========================
# TEST RUNNER
# =========================

if __name__ == "__main__":

    evaluator = ModelEvaluator()

    results = evaluator.run()

    print("\nMODEL COMPARISON:\n")

    for model in results:

        print(model)

        for metric in results[model]:

            print(
                metric,
                round(results[model][metric],3)
            )

        print()