# README – Biomedical Abbreviation Expansion using BERT-based Models

## Overview
This project aims to identify and expand biomedical abbreviations using transformer-based models, specifically focusing on clinical and scientific texts. It includes model training, evaluation, and a Gradio web interface for live user interaction.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Models Used](#models-used)
- [Interface Features](#interface-features)
- [Logging](#logging)
- [Contributors](#contributors)

## Project Structure
```
├── combined_coursework_updated.ipynb   # Training and evaluation of models
├── Frontend.py                         # Gradio app for interactive demo
├── log.csv                             # Stores user input and model output
├── README.md                           # Documentation
```

## Setup Instructions

1. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

2. Install required Python packages:
```bash
pip install torch transformers gradio pandas
```

3. Ensure the following model folders are present locally:
```
SECTION 2.3 – Transformer Fine-tuning/
├── BERT/
│   └── bert_base_cased/model/
├── Bio_ClinicalBERT/
│   └── bio_clinicalbert/model/
```

⚠️ These are locally fine-tuned models. Make sure to avoid fetching from HuggingFace directly.

## How to Use

### Notebook (Model Training & Evaluation)
- Run `combined_coursework_updated.ipynb` step-by-step:
  - Loads and processes PLOD-CW-25 dataset
  - Tokenizes using ClinicalBERT tokenizer
  - Trains using HuggingFace Trainer
  - Saves metrics, plots loss/F1 curves, confusion matrices
  - Outputs per-class performance metrics (Precision, Recall, F1)

### Frontend App (User Interface)
To launch the web interface:

```bash
python Frontend.py
```

This opens a local Gradio interface where users:
- Input a sentence
- Choose between BERT-base-cased with Focal Loss or Bio_ClinicalBERT
- See word-level predictions like:
```
MRI -> B-AC
scan -> O
showed -> O
tumor -> B-LF
```

## Models Used

### Baseline Models:
- Conditional Random Field (CRF - LBFGS)
  - Simple feature-based model.
  - Macro-F1: 0.68, Time: 55ms

- BiLSTM + FastText (100d vectors)
  - Sequence model with pretrained embeddings.
  - Macro-F1: 0.72, Time: 210ms

### Transformer Models:
- Bio_ClinicalBERT
  - Fine-tuned on PLOD-CW-25 dataset.
  - Best performance: Macro-F1: 0.88, Time: 1860ms

- BERT-base-cased (with Focal Loss)
  - Used to handle class imbalance.
  - Performance competitive with good F1 on rare classes.

All models were evaluated using macro F1-score on the test set.

## Interface Features

- Uses Gradio to create a minimal web UI.
- Model selection and token-level output shown.
- Text preprocessing includes:
  - Removing special characters
  - Token splitting
- Real-time prediction with CPU/GPU fallback.

## Logging

Each interaction is saved into `log.csv` with:
- Timestamp
- User input
- Model selected
- Model prediction

This makes it easy to track usage and analyze model behavior.

## Contributors

| Name                                 | Student ID |
|--------------------------------------|------------|
| Tamilkumaran Parivallal Vanitha      | 6899763    |
| Rajgiran Chandrasekar                | 6848530    |
| Vishal Ranganatha                    | 6896942    |
| Akhil Makeswaran                     | 6901051    |
