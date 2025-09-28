## Making the Predictive app for customer churn
## UTF-8
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Validmodel.sav')

# Load CSV
df_1 = pd.read_csv("Notebook EDA\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Define fields
field_names = ["SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender", "Partner",
               "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod", "tenure"]

# Function to preprocess & train model if not found
def train_and_save_model():
    df = df_1.copy()

    # tenure grouping
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df['tenure_group'] = pd.cut(df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df.drop(columns=['tenure'], inplace=True)

    # Target column
    y = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)   # Adjust if already numeric
    X = pd.get_dummies(df.drop(columns=['Churn']))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as Validmodel.sav")
    return model

# Try loading model, else train it
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    else:
        model = train_and_save_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = train_and_save_model()


@app.route("/")
def loadPage():
    form_data = {f'query{i}': "" for i in range(1, 20)}
    return render_template('Codehome.html', field_names=field_names, output1="", output2="", **form_data)


@app.route("/", methods=['POST'])
def predict():
    try:
        input_data = [request.form.get(f'query{i}', '') for i in range(1, 20)]
        form_data = {f'query{i}': input_data[i-1] for i in range(1, 20)}

        # New input row
        new_df = pd.DataFrame([input_data], columns=field_names)
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # tenure preprocessing
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], inplace=True)

        # dummy encoding
        df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                          'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

        if hasattr(model, 'feature_names_in_'):
            df_dummies = df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

        features = df_dummies.to_numpy()

        # Prediction
        single = model.predict(features[-1].reshape(1, -1))[0]
        probability = model.predict_proba(features[-1].reshape(1, -1))[:, 1][0]

        if single == 1:
            o1 = "This customer is likely to be churned!!"
            o2 = f"Confidence: {probability*100:.2f}%"
        else:
            o1 = "This customer is likely to stay!!"
            o2 = f"Confidence: {probability*100:.2f}%"

        return render_template('Codehome.html', output1=o1, output2=o2, field_names=field_names, **form_data)

    except Exception as e:
        form_data = {f'query{i}': request.form.get(f'query{i}', '') for i in range(1, 20)}
        return render_template('Codehome.html',
                               output1=f"Error: {str(e)}",
                               output2="Please check the input data",
                               field_names=field_names,
                               **form_data)


if __name__ == "__main__":
    app.run(debug=True)
