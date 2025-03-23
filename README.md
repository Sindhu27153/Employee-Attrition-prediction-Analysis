# Employee Attrition Analysis and Prediction

## Project Overview
This project aims to analyze employee data to identify key factors influencing attrition and develop a predictive model to anticipate employee departures. The model will assist HR professionals in making data-driven decisions to improve employee retention.

## Features
- **Exploratory Data Analysis (EDA):** Gain insights into attrition trends and influencing factors.
- **Feature Engineering:** Transform raw data into meaningful features for improved model performance.
- **Machine Learning Models:** Train multiple models and select the best-performing one.
- **Model Evaluation:** Assess models using accuracy, precision, recall, and F1-score.
- **Streamlit App:** Deploy an interactive dashboard for real-time predictions.

## Technologies Used
- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn, Streamlit, pickle)
- **Machine Learning Algorithms** (Logistic Regression, Decision Trees, Random Forest, XGBoost, etc.)
- **Deployment:** Streamlit

## Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the trained model and make predictions:
   ```python
   import pickle
   import pandas as pd

   # Load the model
   with open('models/best_lgb_model.pkl', 'rb') as file:
       model = pickle.load(file)

   # Example prediction
   sample_data = pd.DataFrame({...})  # Replace with actual data
   prediction = model.predict(sample_data)
   print(prediction)
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the displayed URL in your browser to use the prediction dashboard.

## Project Structure
```
employee-attrition-prediction/
│── data/                  # Dataset files
│── notebooks/             # Jupyter notebooks for EDA & model training
│── models/                # Saved ML models (Pickle format)
│── app.py                 # Streamlit app
│── train_model.py         # Model training script
│── requirements.txt       # Project dependencies
│── README.md              # Project documentation
```

## Evaluation Metrics
The models are evaluated based on:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## Future Enhancements
- Improve feature engineering
- Implement deep learning models
- Enhance the UI of the Streamlit app
- Automate data ingestion

## Contributors
- Sindhuja Seenivasan

## Acknowledgments
Special thanks to open-source contributors and research papers that guided the development of this project.

