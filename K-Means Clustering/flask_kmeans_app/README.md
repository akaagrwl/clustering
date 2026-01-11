# K-Means Clustering Flask App

This folder contains a minimal Flask application that serves a trained K-Means clustering model and assigns new points to clusters.

Files:
- `train_kmeans.py` (at project root `K-Means Clustering`) — trains and saves `kmeans_model.pkl` containing a dict with `model`, `X_train`, `y_train`.
- `flask_kmeans_app/app.py` — Flask application that loads the pickled model data and exposes endpoints.
- `flask_kmeans_app/templates/index.html` — simple UI to test predictions interactively.
- `flask_kmeans_app/Dockerfile` — Dockerfile to containerize the app.
- `flask_kmeans_app/requirements.txt` — Python dependencies.

Pickle format:
The model is saved as a dictionary so the Flask app can compute cluster centers from the training labels if needed.

```python
model_data = {
  'model': kmeans,      # trained sklearn KMeans estimator
  'X_train': X,         # numpy array of training features
  'y_train': y_km       # numpy array of cluster labels
}
```

This dictionary is written to `kmeans_model.pkl` using `pickle.dump(model_data, file)`.

Usage:
1. Train the model:
```bash
python train_kmeans.py
```
This creates `K-Means Clustering/Python/kmeans_model.pkl` and copies one to `K-Means Clustering/flask_kmeans_app/kmeans_model.pkl`.

2. Run the Flask app (local):
```bash
cd "K-Means Clustering/flask_kmeans_app"
python app.py
```
Open `http://localhost:5000` to use the UI.

3. Build Docker image:
```bash
cd "K-Means Clustering/flask_kmeans_app"
docker build -t kmeans-flask-app .
```

4. Run container:
```bash
docker run -p 5000:5000 kmeans-flask-app
```

Notes:
- The pickle contains training data so the serving code can compute cluster centers.
- For production, prefer using a robust WSGI server (gunicorn) and mount the model file externally or provide it via CI/CD.
