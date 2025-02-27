# Automate ML Model Training

## Project Overview
**Automate ML Model Training** is a Streamlit-based web application designed to simplify the process of training and evaluating machine learning models. The application allows users to upload datasets, preprocess data, choose different ML models, and automatically train and evaluate them.

## Key Features

- **Dataset Upload:** Supports CSV and Excel file formats.

- **Data Preprocessing:**
  - Handles missing values using appropriate imputation techniques.
  - Scales numerical data with options like `StandardScaler`, `MinMaxScaler`, and `RobustScaler`.
  - Encodes categorical data using One-Hot Encoding.

- **Model Selection:** Users can choose between `Logistic Regression`, `Random Forest`, and `SVM`.

- **Training and Evaluation:**
  - Trains the selected model on the processed dataset.
  - Saves the trained model for future use.
  - Evaluates model accuracy using test data.

- **User-Friendly Interface:** Built with Streamlit for an interactive experience.

## How It Works

1. Upload a dataset (`CSV` or `Excel` format).
2. Select the target column and choose a scaling method.
3. Choose a machine learning model to train.
4. Enter a name for the model.
5. Click the **"Train Model"** button to start training.
6. View the model's accuracy after training.

## Technologies Used

- **Python**
- **Streamlit** (for the web interface)
- **scikit-learn** (for data preprocessing and machine learning models)
- **Pandas** (for data manipulation)
- **Pickle** (for saving trained models)



