from flask import Flask, render_template, request
import pickle
import os

# Optional imports; server will fall back to heuristic predictor if sklearn
try:
    import numpy as np
except Exception:
    np = None

app = Flask(__name__)
MODEL_PATH = os.path.join('models', 'best_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
LE_PATH = os.path.join('models', 'label_encoder.pkl')

# Load model artifacts lazily
model = None
scaler = None
le = None
model_loaded = False


def load_artifacts():
    global model, scaler, le
    if model is None:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if scaler is None:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    if le is None:
        with open(LE_PATH, 'rb') as f:
            le = pickle.load(f)
    # mark success
    global model_loaded
    model_loaded = True


def heuristic_predict(features):
    # features: [N,P,K,temperature,humidity,ph,rainfall]
    N, P, K, temperature, humidity, ph, rainfall = features
    if rainfall > 200 and temperature > 24:
        return 'rice'
    if N > 90 and P > 50 and K > 40:
        return 'wheat'
    if ph < 5.5 and rainfall < 80:
        return 'potato'
    if temperature > 30 and rainfall < 60:
        return 'cotton'
    if 6.0 <= ph <= 7.5 and 50 < rainfall < 150:
        return 'maize'
    return 'barley'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        vals = ['N','P','K','temperature','humidity','ph','rainfall']
        data = []
        for v in vals:
            value = request.form.get(v)
            if value is None or value.strip() == '':
                return render_template('result.html', error=f'Missing value for {v}')
            try:
                data.append(float(value))
            except ValueError:
                return render_template('result.html', error=f'Invalid numeric value for {v}')

        # Try to load artifacts; if loading fails, use heuristic
        try:
            load_artifacts()
        except Exception:
            pass

        if model_loaded and np is not None:
            arr = np.array(data).reshape(1, -1)
            arr_s = scaler.transform(arr)
            pred = model.predict(arr_s)
            crop = le.inverse_transform(pred)[0]
            return render_template('result.html', prediction=crop)
        else:
            # Fallback heuristic prediction
            crop = heuristic_predict(data)
            return render_template('result.html', prediction=crop)
    except FileNotFoundError:
        return render_template('result.html', error='Model files not found. Please run train.py first.')
    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
