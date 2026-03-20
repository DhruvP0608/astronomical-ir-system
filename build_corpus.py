import pandas as pd
import os

# Base folder where your CSV files are stored
base_path = os.path.join("CSV Files")

files = [
    ("Solar Eclipse", "SolarEclipseMaster.csv"),
    ("Lunar Eclipse", "LunarEclipseMaster.csv"),
    ("Meteor Shower", "MeteorShowersMaster.csv"),
    ("Planetary Transit", "MercuryTransitsMaster.csv"),
    ("Planetary Conjunction", "PlanetaryConjunctionsMaster.csv")
]

all_docs = []
doc_id = 1

for category, filename in files:
    file_path = os.path.join(base_path, filename)

    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        all_docs.append([
            doc_id,
            category,
            row["event_name"],
            row["description"],
            row["date"],
            row["year"],
            row["month"],
            row["visibility_regions"]
        ])
        doc_id += 1

corpus = pd.DataFrame(all_docs, columns=[
    "doc_id",
    "event_category",
    "event_name",
    "description",
    "date",
    "year",
    "month",
    "visibility_regions"
])

corpus.to_csv("astronomical_corpus.csv", index=False)

print("Unified corpus created successfully.")
