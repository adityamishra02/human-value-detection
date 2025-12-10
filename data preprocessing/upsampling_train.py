import pandas as pd
from sklearn.utils import resample
import numpy as np

# ==========================
# CONFIGURATION
# ==========================
input_path = "training-english/final_training.tsv"          
output_path = "train_upsampled.tsv"
random_state = 42

# ==========================
# LOAD DATA (TSV)
# ==========================
df = pd.read_csv(input_path, sep='\t')

# Exclude ID and text columns — assume all others are label columns
non_label_cols = ["Text-ID", "Sentence-ID", "Text"]
label_cols = [col for col in df.columns if col not in non_label_cols]

print(f"Detected {len(label_cols)} label columns.")

# ==========================
# CHECK CLASS DISTRIBUTION
# ==========================
label_counts = {col: (df[col] == 1).sum() for col in label_cols}
print("\nOriginal label counts:")
for k, v in label_counts.items():
    print(f"{k}: {v}")

# ==========================
# UPSAMPLING PROCESS
# ==========================
max_count = max(label_counts.values())
df_upsampled = df.copy()

for col, count in label_counts.items():
    if count < max_count:
        minority = df[df[col] == 1]
        if len(minority) == 0:
            continue
        
        # sample with replacement to match max count
        samples = resample(minority,
                           replace=True,
                           n_samples=max_count - count,
                           random_state=random_state)
        df_upsampled = pd.concat([df_upsampled, samples], ignore_index=True)
        print(f"Upsampled '{col}' from {count} → {max_count}")

# Shuffle the dataset after upsampling
df_upsampled = df_upsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

# ==========================
# VERIFY BALANCE
# ==========================
balanced_counts = {col: (df_upsampled[col] == 1).sum() for col in label_cols}
print("\nBalanced label counts:")
for k, v in balanced_counts.items():
    print(f"{k}: {v}")

# ==========================
# SAVE RESULT (TSV)
# ==========================
df_upsampled.to_csv(output_path, sep='\t', index=False)
print(f"\n✅ Upsampled dataset saved to: {output_path}")
