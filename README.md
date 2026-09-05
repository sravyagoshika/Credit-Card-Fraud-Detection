💳 Credit Card Fraud Detection System

Machine Learning based system for detecting potentially fraudulent financial transactions.


📌 Project Overview

The Credit Card Fraud Detection System is a Machine Learning based application designed to identify potentially fraudulent financial transactions.

The system analyzes transaction and customer-related information and classifies transactions as either legitimate or fraudulent. The project implements a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, class imbalance handling, model training, hyperparameter tuning, model evaluation, threshold tuning, model explainability, and deployment.

The trained model is exposed through a FastAPI backend and integrated with a Streamlit web application for interactive fraud prediction.

🎯 Objectives


Analyze financial transaction data and identify fraud patterns.
Preprocess and prepare transaction data for Machine Learning.
Perform exploratory data analysis.
Engineer meaningful transaction and customer behavior features.
Handle class imbalance using SMOTE.
Train and compare multiple Machine Learning algorithms.
Perform hyperparameter tuning.
Evaluate models using appropriate fraud detection metrics.
Select an appropriate prediction threshold.
Provide model explanations using SHAP.
Deploy the trained model using FastAPI.
Provide an interactive prediction interface using Streamlit.


📊 Dataset

The project uses a financial transaction dataset containing 10,000 transactions.

Dataset Features

| Feature | Description |
|---|---|
| transaction_id | Unique identifier for each transaction |
| timestamp | Date and time of the transaction |
| amount | Transaction amount |
| merchant_category | Category of the merchant |
| customer_id | Unique identifier of the customer |
| customer_age | Age of the customer |
| customer_location | Location of the customer |
| device_type | Device used for the transaction |
| previous_transactions | Number of previous transactions |
| is_fraud | Target variable: 0 for legitimate and 1 for fraudulent |

Target Variable

is_fraud

- `0` - Legitimate transaction
- `1` - Fraudulent transaction

The dataset contains significantly fewer fraudulent transactions than legitimate transactions, making class imbalance an important consideration.

🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning |
| Imbalanced-learn | SMOTE implementation |
| XGBoost | Gradient boosting model |
| LightGBM | Gradient boosting model |
| SHAP | Model explainability |
| Matplotlib | Data visualization |
| FastAPI | Backend API |
| Uvicorn | API server |
| Streamlit | Web application |

🤖 Machine Learning Models

The following Machine Learning algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The models were compared using precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.

🔄 Project Workflow

```text
Raw Transaction Data
        |
        v
Data Preprocessing
        |
        v
Exploratory Data Analysis
        |
        v
Feature Engineering
        |
        v
Train-Test Split
        |
        v
Class Imbalance Handling using SMOTE
        |
        v
Model Training
        |
        v
Hyperparameter Tuning
        |
        v
Model Evaluation
        |
        v
Threshold Tuning
        |
        v
SHAP Explainability
        |
        v
FastAPI Deployment
        |
        v
Streamlit Application
```

🧹 Data Preprocessing

The preprocessing stage prepares the raw transaction data for Machine Learning.

The process includes:

- Loading the dataset.
- Checking the structure of the dataset.
- Handling categorical features.
- Encoding categorical variables.
- Preparing numerical features.
- Splitting the data into training and testing datasets.
- Scaling required numerical features.

Processed datasets are stored in:

```text
data/processed/
```

📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the transaction data and identify potential fraud patterns.

The project includes visualizations for:

- Transaction amount distribution
- Fraud distribution
- Fraud by merchant category
- Fraud by device type
- Confusion matrices
- SHAP feature importance
- SHAP summary

Generated visualizations are stored in:

```text
reports/figures/
```

⚙️ Feature Engineering

Feature engineering is performed to extract additional information from the transaction data.

Time-Based Features

The transaction timestamp is used to derive:

- Transaction hour
- Day
- Month
- Day of week
- Weekend indicator
- Night-time indicator

Customer Behaviour Features

Customer transaction history is used to derive:

- Average transaction amount
- Transaction amount deviation
- Transaction amount ratio

Categorical transaction information is also transformed into Machine Learning compatible features.

⚖️ Handling Class Imbalance

Fraudulent transactions represent a minority class in the dataset.

To address this class imbalance, the project uses:

SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generates synthetic samples for the minority class and helps Machine Learning models learn patterns associated with fraudulent transactions.

SMOTE-processed datasets are stored in:

```text
data/processed/smote/
```

🧠 Model Training

Six Machine Learning models are trained using the processed dataset:

```text
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost
LightGBM
```

The trained models and supporting files are stored in:

```text
models/
```

🎛️ Hyperparameter Tuning

Hyperparameter tuning was performed to improve the performance of selected Machine Learning models.

Tuned models include:

- Logistic Regression
- Decision Tree
- XGBoost
- LightGBM

The tuned models are stored in:

```text
models/tuned/
```

The tuning results are available in:

```text
reports/metrics/tuned_model_comparison.csv
```

📋 Model Evaluation

Because fraud detection is an imbalanced classification problem, accuracy alone is not sufficient for evaluating the models.

The following metrics are used:

| Metric | Description |
|---|---|
| Precision | Proportion of predicted fraud transactions that are actually fraudulent |
| Recall | Proportion of actual fraud transactions correctly detected |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| PR-AUC | Measures performance with emphasis on the minority class |
| Confusion Matrix | Shows correct and incorrect classification results |

The evaluation results are stored in:

```text
reports/metrics/
```

Confusion matrix visualizations are stored in:

```text
reports/figures/
```

🎚️ Prediction Threshold Tuning

Fraud detection requires careful selection of a prediction threshold because missing a fraudulent transaction can be more significant than generating a false alert.

The deployed application uses a fraud probability threshold of:

```text
0.05
```

Prediction logic:

```text
Probability >= 0.05  →  Fraudulent Transaction
Probability < 0.05   →  Legitimate Transaction
```

Threshold comparison results are available in:

```text
reports/metrics/threshold_comparison.csv
```

🔍 Model Explainability

SHAP (SHapley Additive exPlanations) is used to interpret Machine Learning predictions.

SHAP helps identify the features that have the greatest influence on the model's fraud predictions.

Generated SHAP results include:

```text
reports/figures/shap_feature_importance.png
reports/figures/shap_summary.png
reports/metrics/shap_feature_importance.csv
```

🚀 FastAPI Deployment

The trained fraud detection model is deployed through a FastAPI backend.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks whether the API is running |
| POST | `/predict` | Predicts whether a transaction is fraudulent |

The FastAPI application is located at:

```text
api/main.py
```

Start FastAPI:

```bash
uvicorn api.main:app --reload
```

API address:

```text
http://127.0.0.1:8000
```

🖥️ Streamlit Application

A Streamlit web application provides an interactive interface for fraud prediction.

The user can enter:

- Transaction ID
- Customer ID
- Transaction amount
- Customer age
- Previous transactions
- Timestamp
- Merchant category
- Customer location
- Device type

The Streamlit application sends the transaction information to the FastAPI backend and displays the prediction and fraud probability.

The Streamlit application is located at:

```text
app/streamlit_app.py
```

Start Streamlit:

```bash
streamlit run app/streamlit_app.py
```

📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── financial_fraud_dataset.csv
│   ├── engineered_fraud_dataset.csv
│   └── processed/
│
├── models/
│   └── tuned/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── smote.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── model_evaluation.py
│   ├── threshold_tuning.py
│   └── explainability.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

📦 Installation

Clone the repository:

```bash
git clone https://github.com/sravyagoshika/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

▶️ Running the Application

Start the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit interface can then be used to submit transaction information and obtain a fraud prediction.

📊 Results

The project evaluates multiple Machine Learning models using appropriate classification metrics.

Model comparison:

```text
reports/metrics/model_comparison.csv
```

Tuned model comparison:

```text
reports/metrics/tuned_model_comparison.csv
```

Threshold tuning:

```text
reports/metrics/threshold_comparison.csv
```

SHAP feature importance:

```text
reports/metrics/shap_feature_importance.csv
```git push

✅ Advantages

- Complete end-to-end Machine Learning workflow.
- Handles class imbalance using SMOTE.
- Compares multiple Machine Learning algorithms.
- Includes hyperparameter tuning.
- Uses fraud-specific evaluation metrics.
- Includes prediction threshold tuning.
- Provides model explainability using SHAP.
- Provides a REST API using FastAPI.
- Provides an interactive web interface using Streamlit.
- Includes generated reports, metrics, and visualizations.

⚠️ Limitations

- The dataset is a project-level financial transaction dataset.
- Fraudulent transactions are significantly less frequent than legitimate transactions.
- Model performance depends on the quality and representativeness of the available data.
- Real-world production deployment would require continuous monitoring and periodic model retraining.

🔮 Future Enhancements

- MLflow experiment tracking
- Docker-based deployment
- Real-time transaction streaming using Kafka
- Data drift monitoring
- Automated model retraining
- Feature store integration
- Cloud deployment using AWS, Azure, or Google Cloud
- Integration with real-world payment processing systems

🏁 Conclusion

The Credit Card Fraud Detection System demonstrates the application of Machine Learning techniques to identify potentially fraudulent financial transactions.

The project implements the major stages of a Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, class imbalance handling, model training, hyperparameter tuning, evaluation, threshold tuning, explainability, and deployment.

The final system combines a FastAPI backend with a Streamlit interface to provide an interactive fraud prediction application.

👩‍💻 Author

Sravya

B.Tech - Computer Science Engineering

🔗 Project Repository

Credit Card Fraud Detection on GitHub
