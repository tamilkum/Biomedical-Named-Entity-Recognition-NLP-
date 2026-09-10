# 🧬 Biomedical Abbreviation Expansion using BERT-based Models

A Natural Language Processing project focused on **identifying and expanding biomedical abbreviations** from clinical and scientific text using transformer-based and deep learning models.

The project includes model training and evaluation on the **PLOD-CW-25 dataset**, along with a **Gradio-based interactive interface** for real-time predictions.

---

## 🎯 Project Overview

Biomedical and clinical text contains a large number of abbreviations and domain-specific terminology, making automated information extraction challenging.

This project evaluates multiple approaches for biomedical abbreviation recognition and expansion, ranging from traditional sequence models to transformer-based architectures.

The project explores:

- Biomedical Named Entity Recognition
- Abbreviation identification
- Transformer-based NLP
- Domain-specific language models
- Sequence labeling
- Class imbalance handling
- Model evaluation
- Interactive NLP inference

---

## 🧠 Approach

The project compares traditional deep learning and transformer-based approaches:

```text
                    Biomedical Text
                          ↓
                  Text Preprocessing
                          ↓
                    Tokenization
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
       CRF            BiLSTM +          BERT-based
                     FastText             Models
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ↓
                 Entity / Abbreviation
                      Prediction
                          ↓
                Evaluation & Analysis
                          ↓
                    Macro-F1 Score
```

---

## 📊 Model Performance

The models were evaluated using **Macro-F1** on the test set.

| Model | Macro-F1 | Inference Time |
|---|---:|---:|
| CRF (LBFGS) | 0.68 | 55 ms |
| BiLSTM + FastText | 0.72 | 210 ms |
| **Bio_ClinicalBERT** | **0.88** | 1860 ms |
| BERT-base-cased + Focal Loss | Competitive performance | — |

### Key Result

**Bio_ClinicalBERT achieved the best reported performance with a Macro-F1 score of 0.88**, demonstrating the advantage of domain-specific transformer representations for biomedical text.

---

## 🤖 Models Used

### Baseline Models

**Conditional Random Field (CRF — LBFGS)**

- Feature-based sequence labeling approach
- Macro-F1: **0.68**
- Inference time: **55 ms**

**BiLSTM + FastText**

- Sequential deep learning model
- Uses pretrained 100-dimensional FastText vectors
- Macro-F1: **0.72**
- Inference time: **210 ms**

### Transformer Models

**Bio_ClinicalBERT**

- Fine-tuned on the PLOD-CW-25 dataset
- Domain-specific biomedical language model
- Macro-F1: **0.88**
- Inference time: **1860 ms**

**BERT-base-cased + Focal Loss**

- Transformer-based sequence classification
- Focal Loss used to help address class imbalance
- Competitive performance with improved handling of rare classes

---

## 📚 Dataset

The project uses the **PLOD-CW-25 dataset** for training and evaluation.

The notebook performs:

- Dataset loading and preprocessing
- Tokenization using a ClinicalBERT tokenizer
- Model fine-tuning using Hugging Face Trainer
- Evaluation on the test set
- Precision, Recall, and F1-score analysis
- Loss and F1 learning curves
- Confusion matrix generation

---

## 💻 Interactive Interface

A **Gradio web interface** is included for real-time model inference.

Run the application with:

```bash
python Frontend.py
```

The interface allows users to:

- Enter biomedical text
- Select between available models
- View token-level predictions
- Run predictions using CPU or GPU

Example:

```text
MRI     → B-AC
scan    → O
showed  → O
tumor   → B-LF
```

---

## 📂 Project Structure

```text
Biomedical-Named-Entity-Recognition-NLP-/
│
├── combined_coursework_updated.ipynb
│   └── Model training, evaluation and analysis
│
├── Frontend.py
│   └── Gradio interactive application
│
├── log.csv
│   └── User interaction and prediction logs
│
└── README.md
```

---

## 🛠️ Technologies

`Python` `PyTorch` `Pandas`

`Hugging Face Transformers` `BERT` `Bio_ClinicalBERT`

`BiLSTM` `FastText` `CRF`

`Focal Loss` `Gradio`

`Natural Language Processing` `Biomedical NLP`

---

## 📈 Evaluation

The project evaluates model performance using:

- Macro-F1
- Precision
- Recall
- Per-class F1 scores
- Confusion matrices
- Training and validation curves

Macro-F1 is particularly useful for this task because it provides equal importance to different entity classes, including less frequent classes.

---

## 🔐 Model Files

The fine-tuned BERT and Bio_ClinicalBERT model files are stored locally and are **not included in the repository**.

Expected local structure:

```text
SECTION 2.3 – Transformer Fine-tuning/
├── BERT/
│   └── bert_base_cased/model/
│
└── Bio_ClinicalBERT/
    └── bio_clinicalbert/model/
```

These models must be available locally when running the corresponding notebook or application.

---

## 👥 Contributors

| Name | Student ID |
|---|---|
| Tamilkumaran Parivallal Vanitha | 6899763 |
| Rajgiran Chandrasekar | 6848530 |
| Vishal Ranganatha | 6896942 |
| Akhil Makeswaran | 6901051 |
