# OptiCrop

OptiCrop is an ML-powered crop recommendation system built with Flask.

Quick start

1. Create a Python 3.10+ virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model (generates data/dataset.csv and models/best_model.pkl):

```powershell
python train.py
```

4. Run the Flask app:

```powershell
set FLASK_APP=app.py
flask run
```

Open http://127.0.0.1:5000 in your browser.

Files

- train.py: training script and synthetic dataset generator
- app.py: Flask web app
- templates/: HTML templates
- static/: CSS/JS
- models/: saved scaler and model files

If you want, I can run the training locally now (requires Python and packages).