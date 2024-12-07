from ollama import Client
import os
import json

prompt_template = """
Analyze the following sentence and provide only the emotion scores as a JSON object with the structure below. Include JUST the emotion scores in the JSON object. Do not include any other information in the JSON object. The JSON object should be the only thing in the response.

{{
    "anger": <value>,
    "calmness": <value>,
    "disgust": <value>,
    "eagerness": <value>,
    "fear": <value>,
    "joy": <value>,
    "pleasantness": <value>,
    "sadness": <value>
}}

All the values should be between 0 and 1. The sum of all values MUST be 1.
Remember just to provide the emotion scores in the JSON object. Do not include any other information in the response.
Sentence: "{sentence}"
"""


class MyClient:
    def __init__(self):
        self.client = Client(host=os.getenv(
            'OLLAMA_HOST', 'http://localhost:11434'))
        self.model = os.getenv('OLLAMA_MODEL', 'gemma2')

    def predict_emotion(self, sentence):
        return self.client.chat(model=self.model, messages=[{
            'role': "system",
            'content': prompt_template.format(sentence=sentence)
        }], stream=False)['message']['content']

    def predict_emotion_list(self, sentences):
        return {f"text_{i}": self.predict_emotion(
            text) for i, text in enumerate(sentences)}

    @staticmethod
    def attempt_convert(dict):
        result = {}
        for key, value in dict.items():
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                    result[key] = {k: round(v, 4)
                                   for k, v in result[key].items()}
                    total = sum(result[key].values())
                    if total == 0:
                        raise ValueError("The sum is 0")
                    result[key] = {k: round(v/total, 4)
                                   for k, v in result[key].items()}
                except (json.JSONDecodeError, ValueError, TypeError):
                    result[key] = value
        return result


# if __name__ == '__main__':
#     client = MyClient()
#     sentence = "I am feeling very sad today"
#     response = client.predict_emotion(sentence)
#     print(response['message']['content'])
