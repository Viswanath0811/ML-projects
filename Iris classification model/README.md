Machine Learning- Iris Flower Classification

## Overview

The project constructs a machine learning model to predict iris flowers as one of three species — **Setosa**, **Versicolor**, and **Virginica** — using their petal and sepal measurements.

The model uses the historical Iris dataset and a Random Forest Classifier to classify iris flowers with very high accuracy.

---

## Features

- Loads and preprocesses the Iris dataset
- Splits data into training and test sets
- Performs feature scaling with StandardScaler
- Trains a Random Forest model
- Checks model performance using accuracy score, classification report, and confusion matrix
- Saves the trained model and scaler to use for future predictions
- Includes a sample prediction script to classify new data points

---

## Requirements

- Python 3.7 or later
- Packages specified in `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, seaborn, joblib)

---

## How to Run

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```
# Iris Flower Classification using Machine Learning

## Project Overview

This project builds a machine learning model to classify iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on their petal and sepal measurements.

The model uses the classic Iris dataset and a Random Forest Classifier to achieve high classification accuracy.

---

## Features

- Loads and preprocesses the Iris dataset
- Splits data into training and test sets
- Applies feature scaling with StandardScaler
- Trains a Random Forest model
- Evaluates model performance with accuracy score, classification report, and confusion matrix
- Saves the trained model and scaler for future predictions
- Provides a sample prediction script to classify new data points

---

## Requirements

- Python 3.7 or higher
- Packages listed in `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, seaborn, joblib)

---

## How to Run

1. **Install dependencies:**

  pip install -r requirements.txt

2. **Train the Model**

  python iris_classification.py

3. **Predict New Sample**

  python sample_prediction.py

---

## Example Output:

  Model Accuracy: 96.67%
  Predicted class for sample: setosa


