from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import User_Registration

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.status = 'waiting'  # Default status
            user.save()
            messages.success(request, 'Registration successful. Await admin approval.')
            return redirect('register')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def home(request):
    return render(request, 'base.html')

def userlogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            user = User_Registration.objects.get(username=username, password=password)
            if user.status == 'active':
                return render(request, 'users/userhome.html')
            else:
                messages.error(request, 'Your account is not active. Please wait for admin approval.')
                return redirect('userlogin')
        except User_Registration.DoesNotExist:
            messages.error(request, 'Invalid username or password.')
            return redirect('userlogin')
    return render(request, 'userlogin.html')

def userhome(request):
    return render(request, 'users/userhome.html')


import os
import pandas as pd
import pickle
import joblib
import numpy as np
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = "media/saved_models"
ENCODER_DIR = "media/saved_encoders"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

def preprocess_data(df):
    df = df.dropna()

    # Label encode categorical features
    label_encoders = {}
    for col in ['Country', 'Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        # Save encoder
        with open(os.path.join(ENCODER_DIR, f"{col}_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)

    # Encode target separately
    target_le = LabelEncoder()
    df["Attack Type"] = target_le.fit_transform(df["Attack Type"])
    with open(os.path.join(ENCODER_DIR, "target_encoder.pkl"), "wb") as f:
        pickle.dump(target_le, f)

    X = df.drop("Attack Type", axis=1)
    y = df["Attack Type"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler
    with open(os.path.join(ENCODER_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(request):
    df = pd.read_csv(r"media/augmented_dataset1.csv", low_memory=False)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = round(accuracy * 100, 2)

        # Save the model
        model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

    return render(request, 'users/training.html', {'results': results})


MODEL_DIR = "media/saved_models"
ENCODER_DIR = "media/saved_encoders"

def predict_attack(request):
    if request.method == "POST":
        # Collect user input
        input_data = {
            'Country': request.POST['Country'],
            'Year': float(request.POST['Year']),
            'Target Industry': request.POST['Target Industry'],
            'Financial Loss (in Million $)': float(request.POST['Financial Loss']),
            'Number of Affected Users': float(request.POST['Number of Affected Users']),
            'Attack Source': request.POST['Attack Source'],
            'Security Vulnerability Type': request.POST['Security Vulnerability Type'],
            'Defense Mechanism Used': request.POST['Defense Mechanism Used'],
            'Incident Resolution Time (in Hours)': float(request.POST['Incident Resolution Time'])
        }

        # Load encoders
        encoders = {}
        unseen_flag = False
        for col in ['Country', 'Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']:
            with open(os.path.join(ENCODER_DIR, f"{col}_encoder.pkl"), "rb") as f:
                encoders[col] = pickle.load(f)

            # Handle unseen categories by adding them to the encoder temporarily
            if input_data[col] not in encoders[col].classes_:
                unseen_flag = True
                new_classes = list(encoders[col].classes_) + [input_data[col]]
                encoders[col].classes_ = np.array(new_classes)

            input_data[col] = encoders[col].transform([input_data[col]])[0]

        # Arrange data in correct order
        feature_order = ['Country', 'Year', 'Target Industry', 'Financial Loss (in Million $)',
                         'Number of Affected Users', 'Attack Source', 'Security Vulnerability Type',
                         'Defense Mechanism Used', 'Incident Resolution Time (in Hours)']

        input_values = [input_data[feature] for feature in feature_order]

        # Scale input
        with open(os.path.join(ENCODER_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        input_scaled = scaler.transform([input_values])

        # Load XGBoost model
        model_path = os.path.join(MODEL_DIR, "XGBoost.pkl")
        model = joblib.load(model_path)

        # Predict
        prediction_encoded = model.predict(input_scaled)[0]

        # Decode prediction
        with open(os.path.join(ENCODER_DIR, "target_encoder.pkl"), "rb") as f:
            target_encoder = pickle.load(f)
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]

        return render(request, 'users/predict.html', {
            'prediction': prediction,
            'unseen': unseen_flag
        })

    return render(request, 'users/predict.html')

