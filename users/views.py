from django.shortcuts import render, redirect
from django.contrib import messages

from threat_detection import settings
from .forms import UserRegistrationForm
from django.contrib.auth.hashers import make_password, check_password
from .models import User_Registration

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Hash the password before saving
            user.password = make_password(form.cleaned_data['password'])
            user.status = 'waiting'  # Default status
            user.save()
            messages.success(request, 'Registration successful. Await admin approval.')
            return redirect('userlogin')  # Redirect to login instead of register
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
            user = User_Registration.objects.get(username=username)
            
            # Check password using hash comparison
            if check_password(password, user.password):
                if user.status == 'active':
                    # Store user info in session
                    request.session['user_id'] = user.id
                    request.session['username'] = user.username
                    messages.success(request, f'Welcome, {user.name}!')
                    return redirect('userhome')
                else:
                    messages.error(request, 'Your account is not active. Please wait for admin approval.')
            else:
                messages.error(request, 'Invalid username or password.')
                
        except User_Registration.DoesNotExist:
            messages.error(request, 'Invalid username or password.')
            
        return redirect('userlogin')
        
    return render(request, 'userlogin.html')


def userhome(request):
    # Check if user is logged in
    if 'user_id' not in request.session:
        messages.error(request, 'Please login to access this page.')
        return redirect('userlogin')
    
    # Get user data to display
    try:
        user = User_Registration.objects.get(id=request.session['user_id'])
        return render(request, 'users/userhome.html', {'user': user})
    except User_Registration.DoesNotExist:
        messages.error(request, 'User not found. Please login again.')
        return redirect('userlogin')


def logout(request):
    # Clear session data
    request.session.flush()
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')


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

MODEL_DIR = os.path.join(settings.MEDIA_ROOT, 'saved_models')


def train_models(request):
    """
    Load pre-trained models and evaluate them on test data
    """
    # Load dataset
    csv_path = os.path.join(settings.MEDIA_ROOT, "augmented_dataset1.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Model names matching your saved .pkl files
    model_names = [
        "Logistic_Regression",
        "Decision_Tree",
        "Random_Forest",
        "AdaBoost",
        "SVM",
        "KNN",
        "Naive_Bayes",
        "XGBoost"
    ]

    results = {}
    
    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        
        # Check if model file exists
        if os.path.exists(model_path):
            try:
                # Load the pre-trained model
                model = joblib.load(model_path)
                
                # Make predictions on test data
                preds = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, preds)
                
                # Store results with readable name
                display_name = model_name.replace('_', ' ')
                results[display_name] = round(accuracy * 100, 2)
                
            except Exception as e:
                # Handle any errors loading or predicting
                display_name = model_name.replace('_', ' ')
                results[display_name] = f"Error: {str(e)}"
        else:
            # Model file not found
            display_name = model_name.replace('_', ' ')
            results[display_name] = "Model file not found"

    return render(request, 'users/training.html', {'results': results})


def retrain_models(request):
    """
    Optional function to retrain all models from scratch if needed
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    
    # Load dataset
    csv_path = os.path.join(settings.MEDIA_ROOT, "augmented_dataset1.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(),
        "Random_Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    results = {}
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        
        # Store results
        display_name = name.replace('_', ' ')
        results[display_name] = round(accuracy * 100, 2)

        # Save the model
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

    return render(request, 'users/training.html', {
        'results': results,
        'message': 'Models retrained and saved successfully!'
    })


MODEL_DIR = "media/saved_models"
ENCODER_DIR = "media/saved_encoders"

def predict_attack(request):
    if request.method == "POST":
        # Collect user input (store original values BEFORE any processing)
        original_input = {
            'country': request.POST.get('Country', ''),
            'year': request.POST.get('Year', ''),
            'target_industry': request.POST.get('Target Industry', ''),
            'financial_loss': request.POST.get('Financial Loss', ''),
            'affected_users': request.POST.get('Number of Affected Users', ''),
            'attack_source': request.POST.get('Attack Source', ''),
            'vulnerability_type': request.POST.get('Security Vulnerability Type', ''),
            'defense_mechanism': request.POST.get('Defense Mechanism Used', ''),
            'resolution_time': request.POST.get('Incident Resolution Time', '')
        }
        
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
        
        # DEBUG: Print to console to verify data
        print("Original Input:", original_input)
        print("Prediction:", prediction)
        print("Unseen Flag:", unseen_flag)
        
        context = {
            'prediction': prediction,
            'unseen': unseen_flag,
            'input_data': original_input
        }
        
        return render(request, 'users/predict.html', context)
    
    return render(request, 'users/predict.html')

