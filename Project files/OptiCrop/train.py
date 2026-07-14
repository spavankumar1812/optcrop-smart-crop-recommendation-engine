import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

DATA_PATH = os.path.join("data", "dataset.csv")
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# Generate synthetic dataset
np.random.seed(42)
N = np.random.randint(0, 140, 3000)  # Nitrogen
P = np.random.randint(0, 140, 3000)  # Phosphorous
K = np.random.randint(0, 140, 3000)  # Potassium
temperature = np.round(np.random.normal(25, 6, 3000), 1)
humidity = np.round(np.random.uniform(20, 100, 3000), 1)
ph = np.round(np.random.uniform(4.5, 8.5, 3000), 1)
rainfall = np.round(np.random.uniform(20, 300, 3000), 1)

# Simple heuristic to assign crops
crops = []
for i in range(len(N)):
    if rainfall[i] > 200 and temperature[i] > 24:
        crops.append('rice')
    elif N[i] > 90 and P[i] > 50 and K[i] > 40:
        crops.append('wheat')
    elif ph[i] < 5.5 and rainfall[i] < 80:
        crops.append('potato')
    elif temperature[i] > 30 and rainfall[i] < 60:
        crops.append('cotton')
    elif 6.0 <= ph[i] <= 7.5 and 50 < rainfall[i] < 150:
        crops.append('maize')
    else:
        crops.append('barley')

df = pd.DataFrame({
    'N': N,
    'P': P,
    'K': K,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall,
    'label': crops
})

# Save dataset
df.to_csv(DATA_PATH, index=False)
print(f"Saved synthetic dataset to {DATA_PATH}")

# Features and labels
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

# Label encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Models to compare
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
}

best_score = 0.0
best_name = None
best_model = None

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"{name} accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_name = name
        best_model = model

print("\nBest model:", best_name, "with accuracy", best_score)

# Save best model, scaler and label encoder
with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("Saved best_model.pkl, scaler.pkl, and label_encoder.pkl in models/")

# Optional: show classification report for best model
preds = best_model.predict(X_test_s)
print('\nClassification report for best model:\n')
print(classification_report(y_test, preds, target_names=le.classes_))
