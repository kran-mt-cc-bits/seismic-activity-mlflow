import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# Set the MLflow tracking URI (adjust as needed)
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("Tsunami Prediction Model")

# Load the dataset
data = pd.read_csv('earthquakes.csv')

# Prepare the features and target
features = ['latitude', 'longitude', 'magnitude']
X = data[features]
y = data['tsunami']  # Assuming 1 for tsunami, 0 for no tsunami

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
n_estimators = 100
random_state = 42

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Create and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Print evaluation results
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Log the model
    mlflow.sklearn.log_model(rf_classifier, "random_forest_model")

    # Save the model and feature names locally
    model_data = {
        'model': rf_classifier,
        'features': features
    }
    joblib.dump(model_data, 'tsunami_prediction_model.joblib')

# Function to predict tsunami occurrence
def predict_tsunami(latitude, longitude, magnitude):
    # Load the model and features
    model_data = joblib.load('tsunami_prediction_model.joblib')
    model = model_data['model']
    features = model_data['features']
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[latitude, longitude, magnitude]], 
                            columns=features)
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Get probability scores
    prob = model.predict_proba(input_data)[0]
    
    if prediction[0] == 1:
        return f"Tsunami likely (Probability: {prob[1]:.2%})"
    else:
        return f"Tsunami unlikely (Probability: {prob[0]:.2%})"

# Example usage
lat = 35.652832
lon = 139.839478
mag = 7.5

result = predict_tsunami(lat, lon, mag)
print(f"\nFor earthquake at latitude {lat}, longitude {lon}, magnitude {mag}:")
print(result)

# Additional example predictions
test_cases = [
    (35.652832, 139.839478, 5.5),  # Lower magnitude
    (35.652832, 139.839478, 8.5),  # Higher magnitude
    (0.0, 0.0, 7.5),               # Different location
]

print("\nAdditional test cases:")
for lat, lon, mag in test_cases:
    result = predict_tsunami(lat, lon, mag)
    print(f"Latitude: {lat}, Longitude: {lon}, Magnitude: {mag}")
    print(result)
    print()