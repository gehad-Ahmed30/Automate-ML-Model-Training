import os 
import streamlit as st 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from check import (
    read_data,
    preprocess_data,
    train_model,
    evaluate_model
)

# تحديد المسارات
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

# إعداد الصفحة في Streamlit
st.set_page_config(
    page_title="Automate ML",
    page_icon="🧠",
    layout='centered'
)

st.title("Automate ML Model Training 🧠")

# رفع الملف
uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

if uploaded_file is not None:
    st.markdown("<p style='color: green; font-weight: bold;'>File uploaded successfully ✔</p>", unsafe_allow_html=True)
    
    # قراءة البيانات
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    # قاموس النماذج
    model_dictionary = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "SVM": SVC  
    }

    col1, col2, col3, col4 = st.columns(4)

    scaler_type_list = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']

    with col1:
        target_column = st.selectbox("Select the Target Column", list(data.columns))
    
    with col2:
        scaler_type = st.selectbox("Select a scaler", scaler_type_list)

    with col3:
        selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
    
    with col4:
        model_name = st.text_input("Enter Model name")
    
    if st.button("Train Model"):
        # معالجة البيانات
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column, scaler_type)
        
        # تحويل y_train إلى مصفوفة مناسبة
        y_train = y_train.values.ravel()

        # إنشاء كائن من النموذج المختار
        model_instance = model_dictionary[selected_model]()  

        # تدريب النموذج
        model = train_model(model_instance, X_train, y_train, model_name, parent_dir)

        # تقييم النموذج
        accuracy = evaluate_model(model, X_test, y_test)

        st.success(f"Model trained successfully with accuracy: {accuracy}")
