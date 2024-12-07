from flask import Flask, request, jsonify
from prediction import HandlePrediction
from ollama_model import MyClient
import os
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
tampDet = HandlePrediction()
ollamDet = MyClient()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentences = data['sentences']

    if not sentences or not isinstance(sentences, list):
        return jsonify({"error": "Invalid input. Please provide a list of sentences."}), 400
    try:
        results = tampDet.predict_emotion(sentences)
        return jsonify(results), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred while processing the request."}), 500


@app.route('/predictOllama', methods=['POST'])
def ollamaPredict():
    data = request.get_json()
    sentences = data['sentences']

    if not sentences or not isinstance(sentences, list):
        return jsonify({"error": "Invalid input. Please provide a list of sentences."}), 400
    try:
        results = ollamDet.predict_emotion_list(sentences)
        return jsonify({"results": results,
                        "warning": "The results are not always JSON"
                        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred while processing the request."}), 500


@app.route('/predictAll', methods=['POST'])
def predictAll():
    data = request.get_json()
    sentences = data['sentences']

    if not sentences or not isinstance(sentences, list):
        return jsonify({"error": "Invalid input. Please provide a list of sentences."}), 400
    try:
        # results_tam = handleDet.predict_emotion(sentences)
        # results_ollama = ollamDet.predict_emotion_list(sentences)
        with ThreadPoolExecutor() as executor:
            future_tam = executor.submit(tampDet.predict_emotion, sentences)
            future_ollama = executor.submit(
                ollamDet.predict_emotion_list, sentences)

            results_tam = future_tam.result()
            results_ollama = ollamDet.attempt_convert(future_ollama.result())
        return jsonify({"results_tamp": results_tam,
                        "results_ollama": {
                            # "results": MyClient.attempt_convert(results_ollama),
                            "results": results_ollama,
                            "model": ollamDet.model,
                            "warning": "The results are not always JSON"
                        }
                        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred while processing the request."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG", False))
