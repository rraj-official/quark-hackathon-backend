from flask import Flask, request, jsonify
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

@app.route('/translate', methods=['POST'])
def translate():
    # Get JSON payload
    data = request.get_json()
    if not data or 'q' not in data or 'target' not in data:
        return jsonify({"error": "Missing required parameters 'q' (text) or 'target' (language code)"}), 400

    text = data['q']
    source = data.get('source', 'auto')  # Default to auto-detect
    target = data['target']

    try:
        # Perform translation
        result = translator.translate(text, src=source, dest=target)
        # Return a JSON response similar to LibreTranslate's API
        return jsonify({
            'translatedText': result.text,
            'detectedSourceLanguage': result.src
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
