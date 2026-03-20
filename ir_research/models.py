import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer


# =========================
# CORPUS LOADER
# =========================

class CorpusLoader:

    def __init__(self, path):

        self.df = pd.read_csv(path)

        self.df["doc_id"] = self.df["doc_id"].astype(int)


    def get_documents(self):

        return self.df["description"].tolist()


    def get_doc_ids(self):

        return self.df["doc_id"].tolist()


    def get_tokenized_documents(self):

        docs = self.df["description"].tolist()

        tokenized = [doc.lower().split() for doc in docs]

        return tokenized


    def get_dataframe(self):

        return self.df



# =========================
# TF-IDF MODEL
# =========================

class TFIDFModel:

    def __init__(self, documents, doc_ids):

        self.documents = documents
        self.doc_ids = doc_ids

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.doc_matrix = None


    def build(self):

        self.doc_matrix = self.vectorizer.fit_transform(
            self.documents
        )


    def search(self, query, k=10):

        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vec,
            self.doc_matrix
        )[0]

        top_indices = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_indices:

            results.append(
                (
                    int(self.doc_ids[idx]),
                    float(scores[idx])
                )
            )

        return results



# =========================
# BM25 MODEL
# =========================

class BM25Model:

    def __init__(self, tokenized_documents, doc_ids):

        self.tokenized_documents = tokenized_documents
        self.doc_ids = doc_ids

        self.model = None


    def build(self):

        self.model = BM25Okapi(
            self.tokenized_documents
        )


    def search(self, query, k=10):

        tokenized_query = query.lower().split()

        scores = self.model.get_scores(
            tokenized_query
        )

        scores = np.array(scores)

        top_indices = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_indices:

            results.append(
                (
                    int(self.doc_ids[idx]),
                    float(scores[idx])
                )
            )

        return results



# =========================
# DENSE MODEL (FOR LATER)
# =========================

class DenseModel:

    def __init__(self, documents, doc_ids):

        self.documents = documents
        self.doc_ids = doc_ids

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.embeddings = None


    def build(self):

        self.embeddings = self.model.encode(
            self.documents
        )


    def search(self, query, k=10):

        query_embedding = self.model.encode([query])

        scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indices = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_indices:

            results.append(
                (
                    int(self.doc_ids[idx]),
                    float(scores[idx])
                )
            )

        return results



# =========================
# TEST RUNNER
# =========================

if __name__ == "__main__":

    corpus = CorpusLoader("../astronomical_corpus.csv")

    documents = corpus.get_documents()

    doc_ids = corpus.get_doc_ids()

    tokenized_docs = corpus.get_tokenized_documents()


    print("\nBUILDING TFIDF MODEL...\n")

    tfidf = TFIDFModel(documents, doc_ids)

    tfidf.build()

    tfidf_results = tfidf.search(
        "solar eclipse visible in India",
        5
    )

    print("TFIDF RESULTS:\n")

    for doc_id, score in tfidf_results:

        print(doc_id, score)



    print("\nBUILDING BM25 MODEL...\n")

    bm25 = BM25Model(tokenized_docs, doc_ids)

    bm25.build()

    bm25_results = bm25.search(
        "solar eclipse visible in India",
        5
    )

    print("BM25 RESULTS:\n")

    for doc_id, score in bm25_results:

        print(doc_id, score)