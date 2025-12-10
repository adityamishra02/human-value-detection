# Human Value Detection via Knowledge Distillation (SemEval 2024)

An advanced NLP pipeline designed to detect abstract human values (Freedom, Security, Tradition) in text arguments. This project leverages **Knowledge Distillation** and **Hybrid Data Augmentation** to achieve state-of-the-art performance with a compact model.

## 🧠 Core Architecture
This system uses a **Teacher-Student** framework to compress the knowledge of large Language Models into a deployable format.
- **Teachers:** RoBERTa-Large + DeBERTa-Large (High accuracy, slow inference).
- **Student:** DeBERTa-Base (Optimized for speed/efficiency).
- **Method:** Response-based Knowledge Distillation (KD) to transfer logits from teachers to the student.

## 🛠️ Data Engineering Pipeline
The original dataset suffered from severe class imbalance (rare values had few samples).
1. **Synthetic Augmentation:** Used LLMs to generate synthetic arguments for underrepresented value classes.
2. **Domain Adaptation:** Integrated the **GoEmotions** dataset to enrich the model's understanding of emotional context associated with values.

## 📊 Results
| Model Configuration | F1 Score |
|---------------------|----------|
| RoBERTa-Large (Base)| 0.37     |
| DeBERTa-Large (Base)| 0.38     |
| **Distilled Student** | **0.47** |

**Impact:** Achieved a **27% relative improvement** in F1 score compared to individual large models.

## 🚀 Tech Stack
- **Frameworks:** PyTorch, HuggingFace Transformers
- **Techniques:** Knowledge Distillation, Transfer Learning, Imbalanced Learning
- **Models:** RoBERTa, DeBERTa (v3)
