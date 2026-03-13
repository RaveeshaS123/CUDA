from flask import Flask, render_template, request, jsonify
import torch
from model_logic import predict_clause  # Uses the prediction function we built

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the HTML file in the templates folder
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Perform Inference
        clause_name,confidence = predict_clause(input_text)

        # Return the clause name for the UI
        return jsonify({
            "clause_name": clause_name,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on default port 5000
    app.run(debug=True)
