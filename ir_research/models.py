import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD CORPUS
# =========================

DATA_PATH = "../astronomical_corpus.csv"

df = pd.read_csv(DATA_PATH)

# Build searchable document text
documents = (

    df["event_name"].astype(str) + " " +

    df["event_category"].astype(str) + " " +

    df["description"].astype(str) + " " +

    df["visibility_regions"].astype(str)

).tolist()

doc_ids = df["doc_id"].tolist()


# =========================
# TFIDF BASE MODEL
# =========================

print("\nBUILDING TFIDF MODEL...")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)


# =========================
# TFIDF+ ENHANCED MODEL
# =========================

print("BUILDING TFIDF+ MODEL...")

tfidf_plus_vectorizer = TfidfVectorizer(

    stop_words="english",

    ngram_range=(1,2),

    sublinear_tf=True,

    norm="l2"

)

tfidf_plus_matrix = tfidf_plus_vectorizer.fit_transform(documents)


# =========================
# BM25 MODEL
# =========================

print("BUILDING BM25 MODEL...")

tokenized_docs = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)


# =========================
# DENSE MODEL
# =========================

print("BUILDING DENSE MODEL...")

dense_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

dense_embeddings = dense_model.encode(documents)


print("Models ready.")


# =========================
# SEARCH FUNCTIONS
# =========================

def search_tfidf(query, top_k=5):

    q_vec = tfidf_vectorizer.transform([query])

    scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    ranked = np.argsort(scores)[::-1][:top_k]

    return [(doc_ids[i], scores[i]) for i in ranked]


def search_tfidf_plus(query, top_k=5):

    q_vec = tfidf_plus_vectorizer.transform([query])

    scores = cosine_similarity(q_vec, tfidf_plus_matrix)[0]

    ranked = np.argsort(scores)[::-1][:top_k]

    return [(doc_ids[i], scores[i]) for i in ranked]


def search_bm25(query, top_k=5):

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    ranked = np.argsort(scores)[::-1][:top_k]

    return [(doc_ids[i], scores[i]) for i in ranked]


def search_dense(query, top_k=5):

    q_emb = dense_model.encode([query])

    scores = cosine_similarity(q_emb, dense_embeddings)[0]

    ranked = np.argsort(scores)[::-1][:top_k]

    return [(doc_ids[i], scores[i]) for i in ranked]


# =========================
# TEST RUN
# =========================

if __name__ == "__main__":

    query = "solar eclipse india"

    print("\nTFIDF RESULTS:")
    print(search_tfidf(query))

    print("\nTFIDF+ RESULTS:")
    print(search_tfidf_plus(query))

    print("\nBM25 RESULTS:")
    print(search_bm25(query))

    print("\nDENSE RESULTS:")
    print(search_dense(query))