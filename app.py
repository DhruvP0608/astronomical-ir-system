from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")

# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS saved_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            doc_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("astronomical_corpus.csv")
df["year"] = df["year"].astype(int)

model = SentenceTransformer("all-MiniLM-L6-v2")
event_embeddings = model.encode(df["description"].tolist())

# ---------------- REGION HIERARCHY ----------------
region_hierarchy = {
    "India": ["India", "Asia"],
    "Asia": ["Asia"],
    "Europe": ["Europe"],
    "Africa": ["Africa"],
    "Australia": ["Australia"],
    "North America": ["North America"],
    "South America": ["South America"]
}

# ---------------- HELPERS ----------------
def extract_year(query):
    match = re.search(r"\b(20[2-5][0-9])\b", query)
    return int(match.group()) if match else None

def extract_region(query):
    for region in region_hierarchy.keys():
        if region.lower() in query.lower():
            return region
    return None

def region_match_filter(dataframe, region):
    if not region:
        return dataframe

    visibility_series = dataframe["visibility_regions"].str.lower()
    search_terms = [r.lower() for r in region_hierarchy.get(region, [region])]
    pattern = "|".join(search_terms)

    return dataframe[
        visibility_series.str.contains(pattern, na=False)
        | visibility_series.str.contains("global", na=False)
    ]

def get_user_id():
    if "user" not in session:
        return None

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (session["user"],))
    user = c.fetchone()
    conn.close()

    return user[0] if user else None

def get_saved_doc_ids():
    user_id = get_user_id()
    if not user_id:
        return []

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT doc_id FROM saved_events WHERE user_id = ?", (user_id,))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    return ids

def get_saved_count():
    return len(get_saved_doc_ids())

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        saved_count=get_saved_count(),
        saved_ids=get_saved_doc_ids()
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Registration successful 🌌", "success")
            return redirect(url_for("login"))
        except:
            flash("Username already exists.", "error")
        finally:
            conn.close()

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = user[1]
            flash("Login successful 🌙", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully 🌌", "success")
    return redirect(url_for("home"))

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    event_type_filter = request.form.get("event_type")
    month_filter = request.form.get("month")
    region_filter = request.form.get("region_filter")

    year = extract_year(query)
    region_from_query = extract_region(query)
    region = region_filter if region_filter else region_from_query

    results = []
    suggestion = None

    if year or region or event_type_filter or month_filter:
        results_df = df.copy()

        if year:
            results_df = results_df[results_df["year"] == year]

        if region:
            results_df = region_match_filter(results_df, region)

        if event_type_filter:
            results_df = results_df[results_df["event_category"] == event_type_filter]

        if month_filter:
            results_df = results_df[results_df["month"] == month_filter]

        results = results_df.to_dict(orient="records")
    else:
        query_embedding = model.encode([query])
        scores = cosine_similarity(query_embedding, event_embeddings)[0]
        top_indices = np.argsort(scores)[::-1]
        for idx in top_indices[:20]:
            results.append(df.iloc[idx].to_dict())

    return render_template(
        "index.html",
        results=results,
        suggestion=suggestion,
        saved_ids=get_saved_doc_ids(),
        saved_count=get_saved_count()
    )

@app.route("/save_event", methods=["POST"])
def save_event():
    user_id = get_user_id()
    if not user_id:
        return redirect(url_for("login"))

    doc_id = request.form.get("doc_id")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM saved_events WHERE user_id = ? AND doc_id = ?", (user_id, doc_id))
    exists = c.fetchone()

    if not exists:
        c.execute("INSERT INTO saved_events (user_id, doc_id) VALUES (?, ?)", (user_id, doc_id))
        conn.commit()
        flash("Event saved ⭐", "success")

    conn.close()
    return redirect(url_for("home"))


@app.route("/unsave_event", methods=["POST"])
def unsave_event():
    user_id = get_user_id()
    if not user_id:
        return redirect(url_for("login"))

    doc_id = request.form.get("doc_id")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DELETE FROM saved_events WHERE user_id = ? AND doc_id = ?", (user_id, doc_id))
    conn.commit()
    conn.close()

    flash("Event removed ⭐", "success")
    return redirect(url_for("home"))

@app.route("/saved")
def saved():
    user_id = get_user_id()
    if not user_id:
        return redirect(url_for("login"))

    saved_ids = get_saved_doc_ids()
    saved_events_df = df[df["doc_id"].isin(saved_ids)]
    results = saved_events_df.to_dict(orient="records")

    return render_template(
        "saved.html",
        results=results,
        saved_count=len(saved_ids)
    )
@app.route("/research")
def research():

    with open("ir_research/results.json") as f:
        results = json.load(f)

    for model in results:
        for metric in results[model]:
            results[model][metric] = round(results[model][metric],3)

    return render_template(
    "research.html",
    results=results
)


if __name__ == "__main__":
    app.run(debug=True)
