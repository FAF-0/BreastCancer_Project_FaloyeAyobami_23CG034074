# model_building.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('../Breastcancer/breast_cancer_data.csv')

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Select 5 features from the recommended list
# Selected features: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

X = df[selected_features]
y = df['diagnosis']

print("\nSelected Features:")
print(selected_features)
print("\nTarget Variable Distribution:")
print(y.value_counts())

# Encode target variable (M = Malignant = 1, B = Benign = 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoding: B (Benign) = 0, M (Malignant) = 1")

# Feature scaling (mandatory for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train SVM Model
print("\n" + "="*50)
print("TRAINING SUPPORT VECTOR MACHINE (SVM) MODEL")
print("="*50)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate model
print("\nMODEL EVALUATION METRICS")
print("-" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Save model
model_path = 'breast_cancer_model.pkl'
joblib.dump(svm_model, model_path)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"\n✓ Model saved to '{model_path}'")
print("✓ Scaler saved to 'scaler.pkl'")
print("✓ Label Encoder saved to 'label_encoder.pkl'")

# Demonstrate model reloading and prediction
print("\n" + "="*50)
print("DEMONSTRATING MODEL RELOAD AND PREDICTION")
print("="*50)

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load('scaler.pkl')
loaded_encoder = joblib.load('label_encoder.pkl')

# Test with a sample from test set
test_sample = X_test[0:1]
prediction = loaded_model.predict(test_sample)
prediction_label = loaded_encoder.inverse_transform(prediction)[0]

print(f"\nTest Sample Features (scaled): {test_sample}")
print(f"Predicted Class (encoded): {prediction[0]}")
print(f"Predicted Diagnosis: {prediction_label}")
print(f"Actual Diagnosis: {y.iloc[test_size]}")

print("\n✓ Model successfully reloaded and used for prediction!")