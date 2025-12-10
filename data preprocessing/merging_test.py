import pandas as pd

# === Step 1: Load the TSV files ===
labels_test = pd.read_csv("test-english/labels_test.tsv", sep="\t")
sentences_test = pd.read_csv("test-english/sentences_test.tsv", sep="\t")

# === Step 2: Print sizes before merge ===
print("Labels Test Shape:", labels_test.shape)
print("Sentences Test Shape:", sentences_test.shape)

# === Step 3: Merge using outer join on Text-ID and Sentence-ID ===
merged = pd.merge(
    sentences_test,
    labels_test,
    on=["Text-ID", "Sentence-ID"],
    how="outer"
)

# === Step 4: Print size after merge ===
print("Merged Shape:", merged.shape)

# === Step 5: Save merged output ===
merged.to_csv("merged_test.tsv", sep="\t", index=False)

# === Step 6: Optional — preview first few rows ===
print("\nPreview of merged file:")
print(merged.head())
