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


# Define the paths to the configuration files and the binary model files for each model
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

# Define emotion labels
emotion_labels = ['anger', 'fear', 'sadness', 'calmness',
                  'disgust', 'pleasantness', 'eagerness', 'joy']

# Initialize an empty dictionary to store the models
models = {}

# Load and use each model based on the configuration


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


# Read Configuration from the file
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

# Ensure at least one model is selected
model_names = load_models(instructions)
if not model_names:
    print("No valid models selected. Please check the configuration.")
else:
    # Check if the user provided a file, plaintext, both, or neither
    input_file_name = instructions.get("File", None)
    plaintext = instructions.get("PlainText", None)

    if input_file_name and input_file_name.lower() != 'none':
        # Get the file extension
        file_extension = input_file_name.split('.')[-1].lower()

        # Use the specified file format or print an error if not provided
        if file_extension == 'json':
            with open(input_file_name, 'r') as json_file:
                data = json.load(json_file)
                sentences = data["texts"]
        elif file_extension == 'csv':
            df = pd.read_csv(input_file_name)
            sentences = df["texts"].tolist()
        elif file_extension == 'txt':
            with open(input_file_name, 'r') as txt_file:
                sentences = txt_file.read().splitlines()
        else:
            print(
                f"Error: Unsupported file format '{file_extension}'. The BioEmoDetector accepts only TXT, CSV, or JSON file formats.")
            sentences = []

    elif plaintext and plaintext.lower() != 'none':
        # sentences = plaintext.strip("[]").replace("'", "").split(', ')
        sentences = [sentence.strip()
                     for sentence in plaintext.strip("[]").split("', '")]

    else:
        print("Wrong configuration. Please provide either a file or plaintext.")
        sentences = []

    # Predict for each selected model
    results = {}
    for model_name in model_names:
        model_data = models[model_name]
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        model_results = []
        for input_text in sentences:
            input_text = preprocess_text(input_text)
            # Tokenize the input text
            tokens = tokenizer(input_text, return_tensors='pt',
                               padding=True, truncation=True, max_length=512)

            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Make predictions
            with torch.no_grad():
                outputs = model(**tokens)

            probabilities = torch.sigmoid(outputs.logits)[0]

            # Store the predicted probabilities with emotion labels
            result = {label: probability.item()
                      for label, probability in zip(emotion_labels, probabilities)}
            model_results.append(result)

        model_name = models[model_name]['name']
        results[model_name] = model_results

    majority_results = {i: {label: [] for label in emotion_labels}
                        for i in range(len(sentences))}
    if instructions.get("MajorityVoting", "NO").lower() == 'yes':
        for model_results in results.values():
            for i, result in enumerate(model_results):
                for label, probability in result.items():
                    majority_results[i][label].append(probability)

        majority_sentence_results = {i: {} for i in range(len(sentences))}
        for i, emotion_dict in majority_results.items():
            for label, probabilities in emotion_dict.items():
                majority_prob = sum(probabilities) / len(probabilities)
                majority_sentence_results[i][label] = majority_prob

    # Output results to a text file
    if not os.path.exists('results'):
        os.mkdir('results')
    output_file_name = "Results.txt"
    try:
        with open(output_file_name, 'w') as text_file:
            # Print results for each selected model
            for model_name, model_results in results.items():
                text_file.write(f"\nModel: {model_name}\n")
                for i, result in enumerate(model_results, start=1):
                    text_file.write(f"Prediction for TEXT {i}:\n")
                    for label, probability in result.items():
                        text_file.write(f"{label}: {probability:.4f}\n")

            # Print results for majority voting ensemble if MajorityVoting=YES
            if instructions.get("MajorityVoting", "NO").lower() == 'yes':
                text_file.write("\nResults for Majority Voting Ensemble:\n")
                for i, majority_probs in majority_sentence_results.items():
                    text_file.write(f"Majority Voting for TEXT {i + 1}:\n")
                    for label, majority_prob in majority_probs.items():
                        text_file.write(f"{label}: {majority_prob:.4f}\n")

        print("Using device:", device)
        print(f"Results saved to {output_file_name}")

    except Exception as e:
        print(f"Error: {str(e)}")
