import pandas as pd

# === Step 1: Load the TSV files ===
labels_val = pd.read_csv("validation-english/labels_validation.tsv", sep="\t")
sentences_val = pd.read_csv("validation-english/sentences_validation.tsv", sep="\t")

# === Step 2: Print sizes before merge ===
print("Labels validation Shape:", labels_val.shape)
print("Sentences validation Shape:", sentences_val.shape)

# === Step 3: Merge using outer join on Text-ID and Sentence-ID ===
merged = pd.merge(
    sentences_val,
    labels_val,
    on=["Text-ID", "Sentence-ID"],
    how="outer"
)

# === Step 4: Print size after merge ===
print("Merged Shape:", merged.shape)

# === Step 5: Save merged output ===
merged.to_csv("merged_val.tsv", sep="\t", index=False)

# === Step 6: Optional — preview first few rows ===
print("\nPreview of merged file:")
print(merged.head())
