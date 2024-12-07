import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import json
from huggingface_hub import hf_hub_download
from preprocessing import preprocess_text


class EmotionDetector:
    def __init__(self, requested_models=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.possible_models = ["BioBERT", "BioMedRoBERTa", "BlueBERT",
                                "ClinicalBERT", "CODER", "SciBERT", "ClinicalLongFormer", "MajorityVoting"]

        if not requested_models:
            requested_models = self.possible_models
        elif not any(model in requested_models for model in self.possible_models):
            raise ValueError(
                "No valid models selected. Please check the configuration.")
            return

        self.model_paths = {
            "BioBERT": {
                "config": 'config/BioBERTconfig.json',
                "model": self._download_model("BioBERT.bin"),
                "tokenizer": "dmis-lab/biobert-base-cased-v1.1"
            },
            "BioMedRoBERTa": {
                "config": 'config/BioMedRoBERTaconfig.json',
                "model": self._download_model("BioMedRoBERTa.bin"),
                "tokenizer": "allenai/biomed_roberta_base"
            },
            "BlueBERT": {
                "config": 'config/BlueBERTconfig.json',
                "model": self._download_model("BlueBERT.bin"),
                "tokenizer": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
            },
            "ClinicalBERT": {
                "config": 'config/ClinicalBERTconfig.json',
                "model": self._download_model("ClinicalBERT.bin"),
                "tokenizer": "emilyalsentzer/Bio_ClinicalBERT"
            },
            "CODER": {
                "config": 'config/CODERconfig.json',
                "model": self._download_model("CODER.bin"),
                "tokenizer": "GanjinZero/UMLSBert_ENG"
            },
            "SciBERT": {
                "config": 'config/SciBERTconfig.json',
                "model": self._download_model("SciBERT.bin"),
                "tokenizer": "allenai/scibert_scivocab_cased"
            },
            "ClinicalLongFormer": {
                "config": 'config/ClinicalLongFormerconfig.json',
                "model": self._download_model("ClinicalLongFormer.safetensors"),
                "tokenizer": "yikuan8/Clinical-Longformer"
            }
        }
        self.emotion_labels = ['anger', 'fear', 'sadness',
                               'calmness', 'disgust', 'pleasantness', 'eagerness', 'joy']
        self.models = {}
        self.instructions = self._load_instructions(requested_models)
        self.model_names = self._load_models()

    def _download_model(self, filename):
        return hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename=filename)

    def _load_instructions(self, requested_models):
        return {model: 'yes' if model in requested_models else 'no' for model in self.possible_models}

    def _load_models(self):
        selected_models = []
        for model_name, model_data in self.model_paths.items():
            if self.instructions.get(model_name, 'NO').lower() == 'yes':
                config = AutoConfig.from_pretrained(
                    model_data['config'], local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_data["tokenizer"])
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_data['model'], config=config, local_files_only=True)

                model.to(self.device)

                self.models[model_name] = {
                    "name": model_name,
                    "tokenizer": tokenizer,
                    "model": model
                }

                selected_models.append(model_name)

        return selected_models

    def predict_emotions(self, sentences, output_file=None):
        if not self.model_names:
            print("No valid models selected. Please check the configuration.")
            return

        results_by_text = {f"text_{i}": {} for i in range(len(sentences))}

        for model_name in self.model_names:
            model_data = self.models[model_name]
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]

            for i, input_text in enumerate(sentences):
                input_text = preprocess_text(input_text)
                tokens = tokenizer(input_text, return_tensors='pt',
                                   padding=True, truncation=True, max_length=512)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = model(**tokens)

                probabilities = torch.sigmoid(outputs.logits)[0]
                result = {label: round(probability.item(), 4) for label, probability in zip(
                    self.emotion_labels, probabilities)}

                results_by_text[f"text_{i}"][model_name] = result

        if self.instructions.get("MajorityVoting", "NO").lower() == 'yes':
            for i in range(len(sentences)):
                majority_result = self._calculate_majority_voting(
                    results_by_text[f"text_{i}"])
                results_by_text[f"text_{i}"]["MajorityVoting"] = majority_result

        if output_file:
            self._write_to_json_file(output_file, results_by_text)
        print("Device:", self.device)
        return results_by_text

    def _calculate_majority_voting(self, models_results):
        emotion_probabilities = {label: [] for label in self.emotion_labels}

        for model_result in models_results.values():
            for label, probability in model_result.items():
                emotion_probabilities[label].append(probability)

        majority_result = {}
        for label, probabilities in emotion_probabilities.items():
            majority_prob = sum(probabilities) / len(probabilities)
            majority_result[label] = round(majority_prob, 4)

        return majority_result

    def _write_to_json_file(self, filename, data):
        try:
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data successfully written to {filename}")
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")


# detector = EmotionDetector()
# results = detector.predict_emotions(
#     ['bashar is happy, Jesus is friendly', 'ahmad is sad, and fear'], "results/Results.json")
# print(results)
