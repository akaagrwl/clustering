"""
Flask app to serve K-Means cluster predictions.

- Loads a pickled `model_data` dictionary containing:
  - 'model': KMeans estimator
  - 'X_train': training features
  - 'y_train': training labels

- Computes cluster centers (from saved training labels) and assigns incoming
  points to the nearest center using Euclidean distance.

Endpoints:
- GET `/` -> web UI
- POST `/predict` -> JSON input {income, spending_score} -> returns cluster (1..k)
- GET `/health` -> returns status and whether model/centers are loaded

This file is documented and commented for clarity.
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.spatial.distance import cdist

app = Flask(__name__)

# Storage for model and cluster info
model_data = {}


def calculate_cluster_centers(X, y):
    """Compute mean vector for each cluster label in y."""
    centers = []
    for cluster_id in range(int(y.max()) + 1):
        centers.append(X[y == cluster_id].mean(axis=0))
    return np.vstack(centers)


def load_model():
    """Load pickled model data from multiple possible locations."""
    global model_data
    possible = [
        os.path.join(os.path.dirname(__file__), 'kmeans_model.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'Python', 'kmeans_model.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'Python', 'Mall_Customers', 'kmeans_model.pkl'),
        'kmeans_model.pkl'
    ]
    model_path = None
    for p in possible:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        print("Model file not found. Checked:")
        for p in possible:
            print("  ", p)
        return False

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    # Accept both plain model (legacy) and dict format
    if isinstance(data, dict):
        model_data['model'] = data.get('model')
        model_data['X_train'] = data.get('X_train')
        model_data['y_train'] = data.get('y_train')
    else:
        # legacy: data is just an estimator
        model_data['model'] = data
        model_data['X_train'] = None
        model_data['y_train'] = None

    # compute centers if training data present
    if model_data.get('X_train') is not None and model_data.get('y_train') is not None:
        model_data['centers'] = calculate_cluster_centers(model_data['X_train'], model_data['y_train'])
    elif hasattr(model_data.get('model'), 'cluster_centers_'):
        # If the estimator has cluster_centers_ (KMeans) use it
        model_data['centers'] = model_data['model'].cluster_centers_
    else:
        model_data['centers'] = None

    print(f"Loaded model from: {model_path}")
    return True


@app.route('/')
def home():
    """Render UI page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Accepts JSON with 'income' and 'spending_score' and returns predicted cluster."""
    try:
        data = request.get_json()
        if not data or 'income' not in data or 'spending_score' not in data:
            return jsonify({'error': 'Provide income and spending_score'}), 400

        income = float(data['income'])
        spending_score = float(data['spending_score'])

        # basic validation
        if income < 0 or spending_score < 0 or spending_score > 100:
            return jsonify({'error': 'Income must be >=0; spending_score in [0,100]'}), 400

        if not model_data or model_data.get('model') is None:
            return jsonify({'error': 'Model not loaded'}), 500

        centers = model_data.get('centers')
        if centers is None:
            return jsonify({'error': 'No cluster centers available; ensure model saved with training data'}), 500

        point = np.array([[income, spending_score]])
        dists = cdist(point, centers, metric='euclidean')[0]
        cluster = int(np.argmin(dists))

        return jsonify({
            'success': True,
            'income': income,
            'spending_score': spending_score,
            'cluster': cluster + 1,
            'message': f'Assigned to cluster {cluster + 1}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Simple health endpoint showing if model and centers present."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_data.get('model') is not None,
        'centers_available': model_data.get('centers') is not None
    })


if __name__ == '__main__':
    ok = load_model()
    if not ok:
        print('Model load failed; starting app anyway (endpoints will return errors).')

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
