from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from scorer import CommunicationScorer

app = Flask(__name__)
CORS(app)

# Initialize the scorer (will load models on startup)
print("Initializing Communication Scorer...")
scorer = CommunicationScorer()
print("✓ Scorer ready!")

@app.route('/')
def home():
    return jsonify({
        'message': 'Communication Skills Scoring API',
        'version': '1.0',
        'endpoints': {
            '/api/score': 'POST - Score a transcript',
            '/api/health': 'GET - Health check'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'scorer_ready': True})

@app.route('/api/score', methods=['POST'])
def score_transcript():
    try:
        # Get transcript from request
        data = request.get_json()
        
        if not data or 'transcript' not in data:
            return jsonify({
                'error': 'Missing transcript in request body'
            }), 400
        
        transcript = data['transcript'].strip()
        
        if not transcript:
            return jsonify({
                'error': 'Transcript cannot be empty'
            }), 400
        
        # Score the transcript
        result = scorer.score_transcript(transcript)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)