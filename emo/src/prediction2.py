import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from huggingface_hub import hf_hub_download
import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
import pandas as pd
import json
from preprocessing import preprocess_text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BioBERT_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="BioBERT.bin")
BioMedRoBERTa_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="BioMedRoBERTa.bin")
BlueBERT_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="BlueBERT.bin")
ClinicalBERT_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="ClinicalBERT.bin")
CODER_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="CODER.bin")
SciBERT_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="SciBERT.bin")
ClinicalLongFormer_path = hf_hub_download(
    repo_id="Bashar-Alshouha/BioEmoDetector", filename="ClinicalLongFormer.safetensors")


model_paths = {
    "BioBERT": {
        "config": 'config/BioBERTconfig.json',
        "model": BioBERT_path,
        "tokenizer": "dmis-lab/biobert-base-cased-v1.1"
    },
    "BioMedRoBERTa": {
        "config": 'config/BioMedRoBERTaconfig.json',
        "model": BioMedRoBERTa_path,
        "tokenizer": "allenai/biomed_roberta_base"
    },
    "BlueBERT": {
        "config": 'config/BlueBERTconfig.json',
        "model": BlueBERT_path,
        "tokenizer": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
    },
    "ClinicalBERT": {
        "config": 'config/ClinicalBERTconfig.json',
        "model": ClinicalBERT_path,
        "tokenizer": "emilyalsentzer/Bio_ClinicalBERT"
    },
    "CODER": {
        "config": 'config/CODERconfig.json',
        "model": CODER_path,
        "tokenizer": "GanjinZero/UMLSBert_ENG"
    },
    "SciBERT": {
        "config": 'config/SciBERTconfig.json',
        "model": SciBERT_path,
        "tokenizer": "allenai/scibert_scivocab_cased"
    },
    "ClinicalLongFormer": {
        "config": 'config/ClinicalLongFormerconfig.json',
        "model": ClinicalLongFormer_path,
        "tokenizer": "yikuan8/Clinical-Longformer"
    }
}
emotion_labels = ['anger', 'fear', 'sadness', 'calmness',
                  'disgust', 'pleasantness', 'eagerness', 'joy']

models = {}


def load_models(instructions):
    selected_models = []
    for model_name, model_data in model_paths.items():
        # Check if the model is selected in the instructions (case-insensitive)
        if instructions.get(model_name, 'NO').lower() == 'yes':
            # Load the configuration
            config = AutoConfig.from_pretrained(
                model_data['config'], local_files_only=True)

            # Load the tokenizer separately
            tokenizer = AutoTokenizer.from_pretrained(model_data["tokenizer"])

            # Load the model weights
            model = AutoModelForSequenceClassification.from_pretrained(
                model_data['model'], config=config, local_files_only=True)

            # Store the loaded model in the dictionary

            model.to(device)

            models[model_name] = {
                "name": model_name,
                "tokenizer": tokenizer,
                "model": model
            }

            selected_models.append(model_name)

    return selected_models


instructions_file_name = "src/Configuration.txt"
with open(instructions_file_name, 'r') as instructions_file:
    # Process instructions into a dictionary
    lines = instructions_file.read().splitlines()
    instructions = {}
    sentences = []
    collecting_sentences = False
    for line in lines:
        if 'Sentences' in line:
            collecting_sentences = True
            continue
        if collecting_sentences and line.strip():
            sentences.append(line.strip())
        elif '=' in line:
            key, value = map(str.strip, line.split('=', 1))
            instructions[key] = value


print("Instructions:", instructions)

model_names = load_models(instructions)


def predict_emotions(sentences: list):
    if not model_names:
        print("No valid models selected. Please check the configuration.")
        return

    # empty dictionary to store the results
    results_by_text = {f"text_{i}": {} for i in range(len(sentences))}

    # pred for each model
    for model_name in model_names:
        model_data = models[model_name]
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        for i, input_text in enumerate(sentences):
            input_text = preprocess_text(input_text)
            tokens = tokenizer(input_text, return_tensors='pt',
                               padding=True, truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Make predictions
            with torch.no_grad():
                outputs = model(**tokens)

            probabilities = torch.sigmoid(outputs.logits)[0]

            result = {label: round(probability.item(), 4)
                      for label, probability in zip(emotion_labels, probabilities)}

            results_by_text[f"text_{i}"][model_name] = result

    if instructions.get("MajorityVoting", "NO").lower() == 'yes':
        for i in range(len(sentences)):
            majority_result = calculate_majority_voting(
                results_by_text[f"text_{i}"])
            results_by_text[f"text_{i}"]["MajorityVoting"] = majority_result

    write_to_json_file("results/Results.json", results_by_text)
    return results_by_text


def calculate_majority_voting(models_results):
    emotion_probabilities = {label: [] for label in emotion_labels}

    for model_result in models_results.values():
        for label, probability in model_result.items():
            emotion_probabilities[label].append(probability)

    majority_result = {}
    for label, probabilities in emotion_probabilities.items():
        majority_prob = sum(probabilities) / len(probabilities)
        majority_result[label] = round(majority_prob, 4)

    return majority_result


def write_to_json_file(filename, data):
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


print(predict_emotions(
    ['bashar is happy, Jesus is friendly', 'ahmad is sad, and fear']))
