import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load saved model and scaler
model = joblib.load('iris_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load iris to get feature names
iris = load_iris()

# New sample to predict
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)

# Scale sample
sample_scaled = scaler.transform(sample)

# Predict class
predicted_class_index = model.predict(sample_scaled)[0]
predicted_class = iris.target_names[predicted_class_index]

print(f"Predicted class for sample: {predicted_class}")