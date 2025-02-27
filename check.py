import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import os
import pickle
import streamlit as st

def read_data(uploaded_file):
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format! Please upload a CSV or Excel file.")
    return None

def preprocess_data(data, target_column, scaler_type):
    x = data.drop(columns=[target_column], axis=1)
    y = data[target_column]

    numerical_cols = x.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # معالجة القيم المفقودة للبيانات العددية
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
        else:
            scaler = None
        
        if scaler:
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # معالجة القيم الفئوية (Categorical)
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        encoded_col_names = encoder.get_feature_names_out(categorical_cols)
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_col_names, index=X_train.index)
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_col_names, index=X_test.index)

        # دمج البيانات المحولة مع الأصلية بعد حذف الأعمدة الفئوية
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

def train_model(model, x_train, y_train, model_name, parent_dir='.'):
    model.fit(x_train, y_train)

    model_dir = os.path.join(parent_dir, "trained_model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pkl")

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return round(accuracy, 2)

