from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from summarizer import (
    summarize_text, get_random_sample, get_dataset_info,
    compute_rouge, evaluate_on_dataset
)
import os

app = Flask(__name__)
CORS(app)

# ── Static file serving ────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve any static file from the project directory."""
    if os.path.exists(os.path.join('.', filename)):
        return send_from_directory('.', filename)
    return 'Not found', 404


# ── API endpoints ──────────────────────────────────────────────────────────

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    num_sentences = data.get('num_sentences', 3)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        summary = summarize_text(text, int(num_sentences))
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/info', methods=['GET'])
def dataset_info():
    """Return dataset split info (article counts)."""
    try:
        info = get_dataset_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/sample', methods=['GET'])
def dataset_sample():
    """Get random article(s) from the dataset."""
    split = request.args.get('split', 'test')
    count = int(request.args.get('count', 1))
    count = min(count, 5)  # Cap at 5
    
    try:
        samples = get_random_sample(split, count)
        if not samples:
            return jsonify({'error': 'Dataset not found or empty'}), 404
        return jsonify({'samples': samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/summarize_sample', methods=['POST'])
def summarize_sample():
    """Summarize a dataset article and compare with reference summary."""
    data = request.get_json()
    article = data.get('article', '')
    highlights = data.get('highlights', '')
    num_sentences = int(data.get('num_sentences', 3))

    if not article:
        return jsonify({'error': 'No article provided'}), 400

    try:
        generated = summarize_text(article, num_sentences)
        result = {'generated_summary': generated}
        
        if highlights:
            scores = compute_rouge(generated, highlights)
            result['reference_summary'] = highlights
            result['rouge_scores'] = scores
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dataset/evaluate', methods=['POST'])
def evaluate():
    """Run evaluation on random dataset samples."""
    data = request.get_json() or {}
    num_samples = min(int(data.get('num_samples', 10)), 50)
    num_sentences = int(data.get('num_sentences', 3))
    split = data.get('split', 'test')

    try:
        results = evaluate_on_dataset(num_samples, num_sentences, split)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
