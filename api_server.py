"""
REST API для классификатора танцевальных движений
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Добавляем пути
sys.path.insert(0, str(Path(__file__).parent / "dance_classifier"))

from dance_classifier.inference.predict import DanceClassifierPredictor

app = Flask(__name__)
CORS(app)  # Разрешает запросы с фронтенда

# Инициализация модели при старте
predictor = None

@app.route('/health', methods=['GET'])
def health():
    """Проверка что сервер работает"""
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/init', methods=['POST'])
def init_model():
    """Инициализация модели"""
    global predictor
    
    data = request.json
    model_path = data.get('model_path', 'best_model.pth')
    metadata_path = data.get('metadata_path', 'metadata.json')
    
    try:
        predictor = DanceClassifierPredictor(
            model_path=model_path,
            metadata_path=metadata_path
        )
        return jsonify({
            "success": True,
            "message": "Model loaded successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/classify', methods=['POST'])
def classify():
    """Классификация движения из poses.jsonl"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "success": False,
            "error": "Model not initialized. Call /init first"
        }), 400
    
    data = request.json
    poses_file = data.get('poses_file')
    
    if not poses_file:
        return jsonify({
            "success": False,
            "error": "poses_file is required"
        }), 400
    
    try:
        result = predictor.predict_from_poses(poses_file)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/classify_multiple', methods=['POST'])
def classify_multiple():
    """Классификация нескольких видео"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "success": False,
            "error": "Model not initialized. Call /init first"
        }), 400
    
    data = request.json
    poses_files = data.get('poses_files', [])
    
    if not poses_files:
        return jsonify({
            "success": False,
            "error": "poses_files array is required"
        }), 400
    
    results = []
    for poses_file in poses_files:
        try:
            result = predictor.predict_from_poses(poses_file)
            result['file'] = poses_file
            results.append(result)
        except Exception as e:
            results.append({
                "file": poses_file,
                "success": False,
                "error": str(e)
            })
    
    return jsonify({"results": results})

if __name__ == '__main__':
    print("=" * 80)
    print("REST API SERVER - Dance Movement Classifier")
    print("=" * 80)
    print("\nEndpoints:")
    print("  GET  /health              - Check server status")
    print("  POST /init                - Initialize model")
    print("  POST /classify            - Classify single video")
    print("  POST /classify_multiple   - Classify multiple videos")
    print("\nServer starting on http://localhost:5000")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)










