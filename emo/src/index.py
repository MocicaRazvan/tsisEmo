from flask import Flask, request, jsonify
from prediction3 import EmotionDetector
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentences = data['sentences']
    requested_models = data.get('requested_models', None)

    if not sentences or not isinstance(sentences, list):
        return jsonify({"error": "Invalid input. Please provide a list of sentences."}), 400
    try:
        detector = EmotionDetector(requested_models)
        results = detector.predict_emotions(sentences)
        return jsonify(results), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred while processing the request."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG", False))
