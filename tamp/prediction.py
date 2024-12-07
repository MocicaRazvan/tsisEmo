from keras.models import load_model
import os
import joblib
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
import re
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import numpy as np
nltk.download('stopwords')


class HandlePrediction:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model = load_model(os.path.join(
            base_dir, "models", "bio_model.h5"))
        self.label_encoder = joblib.load(os.path.join(
            base_dir, "models", "bio_label_encoder.pkl"))
        with open(os.path.join(base_dir, "models", "bio_sizes.json"), 'r') as json_file:
            self.sizes = json.load(json_file)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        with open(os.path.join(base_dir, "models", "bio_tokenizer.json"), 'r') as json_file:
            tokenizer_json = json_file.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)

    def clean_text(self, texts: list):
        dex = []
        for text in texts:
            text = re.sub("[^a-zA-Z0-9]", " ", text).lower().split()
            text = [self.stemmer.stem(word)
                    for word in text if word not in self.stop_words]
            text = ' '.join(text)
            dex.append(text)
        sequences = self.tokenizer.texts_to_sequences(dex)
        padded_sequence = pad_sequences(
            sequences, maxlen=self.sizes['max_length'], padding='pre')
        return padded_sequence

    def predict_emotion(self, texts: list):
        preprocessed_texts = self.clean_text(texts)
        results_by_text = {}
        for i, text in enumerate(preprocessed_texts):
            result = {}
            prediction = self.model.predict(np.array(text).reshape(1, -1))
            for j in range(len(prediction[0])):
                result[self.label_encoder.inverse_transform(
                    [j])[0]] = round(float(prediction[0][j]), 4)
            results_by_text[f"text_{i}"] = result
        return results_by_text


# if __name__ == "__main__":
#     hp = HandlePrediction()
#     texts = ["I am so happy today", "I am so sad today"]
#     print(hp.predict_emotion(texts))
