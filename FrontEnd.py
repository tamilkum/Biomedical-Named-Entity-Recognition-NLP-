# =========================
# Imports
# =========================
import gradio as gr
import torch
import re
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification


# =========================
# Main Analysis Function
# =========================
def analyzeInput(inText: str, model_choice: str) -> str:
    """
    Handles input validation, calls model prediction, formats output, and logs results.
    """
    if not inText or len(inText.strip()) == 0 or not model_choice:
        return "Error: Please provide all inputs (sentence and model)."
    output = modelPrediction(inText, model_choice)
    formatted_output = "\n".join([f"{word} -> {label}" for word, label in output])
    save_to_csv(inText, model_choice, str(output))
    return f"The output of {model_choice}: \n\n {formatted_output}"


# =========================
# Model Prediction Function
# =========================
def modelPrediction(input: str, model_choice: str) -> list:
    """
    Loads the selected model, tokenizes input, predicts labels, and aligns predictions with tokens.
    """
    LABEL_NAMES = ["O", "B-AC", "B-LF", "I-LF"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clean and tokenize input
    text = re.sub(r"[()\[\]{}.,!?;:'\"-]", "", input)
    tokens = text.split()

    # Model selection and loading
    if model_choice == "BERT-base-cased":
        PATH = "SECTION 2.3 – Transformer Fine-tuning/BERT/bert_base_cased/model"
        tokenizer = AutoTokenizer.from_pretrained(PATH)
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        word_ids = encoded.word_ids()
        inputs = {k: v.to(device) for k, v in encoded.items()}
        model = AutoModelForTokenClassification.from_pretrained(PATH).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    elif model_choice == "Bio_ClinicalBERT":
        PATH = "SECTION 2.3 – Transformer Fine-tuning/Bio_ClinicalBERT/bio_clinicalbert/model"
        tokenizer = AutoTokenizer.from_pretrained(PATH)
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        word_ids = encoded.word_ids()
        inputs = {k: v.to(device) for k, v in encoded.items()}
        model = AutoModelForTokenClassification.from_pretrained(PATH).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    else:
        raise ValueError("Invalid model choice.")

    # Align predictions with original tokens
    aligned_preds = []
    prev_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word:
            continue
        aligned_preds.append(LABEL_NAMES[preds[idx]])
        prev_word = word_id

    return list(zip(tokens, aligned_preds))


# =========================
# Logging Function
# =========================
def save_to_csv(input_text: str, model_choice: str, output: str) -> None:
    """
    Saves the input, model choice, output, and timestamp to a CSV log file.
    """
    data = {
        "Date and Time": [datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        "Input": [input_text],
        "Model Choice": [model_choice],
        "Output": output,
    }
    df = pd.DataFrame(data)
    df.to_csv("log.csv", mode="a", header=False, index=False)


# =========================
# Gradio UI Setup
# =========================
inputs = [
    gr.Textbox(
        label="Input",
        placeholder="Enter your sentence here ...",
    ),
    gr.Radio(
        [
            "BERT-base-cased",
            "Bio_ClinicalBERT",
        ],
        label="Model selection",
        info="Choose a model",
    ),
]

title = (
    "<div style='font-size:0.75em; font-weight:bold; border:2px solid orange; padding:10px; border-radius:8px; text-align:center;'>"
    "NATURAL LANGUAGE PROCESSING (COMM061), SEMESTER 2, 2024-25<br>"
    "<span style='font-size:0.65em; font-weight:500;'>"
    "SCHOOL OF COMPUTER SCIENCE & ELECTRICAL ENGINEERING, UNIVERSITY OF SURREY"
    "</span></div>"
)
description = (
    "<div style='margin-left:20px; border-left: 4px solid orange; padding-left: 16px;'>"
    "<b><u>Team Members: <br> </u></b>"
    "1. Tamilkumaran Parivallal Vanitha (6899763)<br>"
    "2. Rajgiran Chandrasekar (6848530)<br>"
    "3. Vishal Ranganatha (6896942)<br>"
    "4. Akhil Makeswaran (6901051)<br>"
    "</div>"
)

Project = gr.Interface(
    fn=analyzeInput,
    inputs=inputs,
    outputs=gr.Textbox(
        label="Output",
        placeholder="Model's output here ...",
    ),
    title=title,
    description=description,
)

# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    Project.launch()
