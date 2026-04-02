# 📌 SmartPolicy

### Insurance Charges Prediction System

**Technology:** Python, Machine Learning, Data Analytics
**Frontend:** Streamlit
**Development Tool:** Jupyter Notebook (Scripts used for production)
**Internship Project Submission**

---

# 📖 1. Project Overview

## 1.1 Introduction

SmartPolicy is a Machine Learning-based insurance charge prediction system that estimates medical insurance costs based on user attributes such as age, BMI, smoking habits, gender, number of children, and region.

The project combines **data science, predictive modeling, and interactive Streamlit UI** to create a real-world insurance analytics application.

---

## 🎯 2. Aim

To develop a machine learning-based web application that accurately predicts insurance charges and provides insights into cost-driving factors.

---

## 🧩 3. Problem Statement

Insurance companies face challenges in accurately estimating medical insurance charges due to multiple influencing factors.
Manual estimation may lead to incorrect pricing, risk imbalance, and loss of profit.

This project aims to build a predictive model to:

* Estimate insurance charges
* Identify significant cost-driving factors
* Provide an interactive prediction interface

---

# 🏗 4. System Architecture

```
Dataset → Data Preprocessing → EDA → Feature Engineering →
Model Training → Model Evaluation → Model Selection →
Model Saving (.pkl) → Streamlit UI → User Prediction
```

---

# 🚀 5. Phase-Wise Project Development

---

# 🔹 Phase 1: Requirement Analysis

### Objectives:

* Understand insurance pricing factors
* Define dataset requirements
* Identify ML algorithms to be used
* Define system workflow

### Deliverables:

* Problem statement
* Dataset description
* Tech stack selection

---

# 🔹 Phase 2: Data Collection

### Dataset Used:

Enhanced Insurance Dataset (10,000 Records)

### Features:

* Age, Gender, BMI, Children, Smoker, Region
* **Heart Rate**
* **Systolic Blood Pressure**
* **Diastolic Blood Pressure**
* **Diabetes (Yes/No)**
* **Hypertension (Yes/No)**
* **Cancer History (Yes/No)**
* Charges (Target Variable)

### Deliverables:

* Clean dataset file (`data/enhanced_insurance_dataset_10k.csv`)
* Data summary report

---

# 🔹 Phase 3: Data Preprocessing

### Steps:

* Handling missing values
* Encoding categorical variables (One-Hot Encoding)
* Feature scaling (StandardScaler)
* Outlier detection

### Tools Used:

* Pandas
* NumPy
* Scikit-learn

### Deliverables:

* Clean processed dataset
* Preprocessing script (`src/train_model.py`)

---

# 🔹 Phase 4: Exploratory Data Analysis (EDA)

### Analysis Performed:

* Correlation Heatmap
* Age vs Charges
* BMI vs Charges
* Smoking impact on charges
* Regional distribution analysis

### Visualization Tools:

* Matplotlib (Integrated in analysis)
* Seaborn (Integrated in analysis)

### Key Insights:

* Smokers have significantly higher charges
* Age and BMI positively correlate with insurance cost
* Smoking is the strongest cost-driving factor

### Deliverables:

* EDA Report
* Graph visualizations

---

# 🔹 Phase 5: Model Building

### Algorithms Used:

* Random Forest Regressor

### Process:

* Train-test split (80-20)
* Model training
* Hyperparameter tuning (n_estimators=100)

### Evaluation Metrics:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

### Deliverables:

* Model comparison table
* Best performing model selection

---

# 🔹 Phase 6: Model Evaluation & Selection

After testing, Random Forest Regressor was selected.

✅ Best Model Selected: Random Forest Regressor

Reason:

* Higher accuracy (R² ~0.84)
* Better generalization
* Handles nonlinear relationships effectively

---

# 🔹 Phase 7: Model Deployment Preparation

### Steps:

* Save trained model using Pickle

```python
import pickle
pickle.dump(model, open("models/smartpolicy_model.pkl", "wb"))
```

* Save encoder/scaler pipeline
* Create requirements.txt

### Deliverables:

* models/smartpolicy_model.pkl
* requirements.txt

---

# 🔹 Phase 8: Streamlit Web Application Development

### UI Features:

* User input form
* Real-time prediction
* Clean dashboard design
* Risk insights display

### Streamlit Components:

* st.title()
* st.number_input()
* st.selectbox()
* st.button()
* st.success()

### Workflow:

* User Input → Model → Prediction → Display Result

---

# 🔹 Phase 9: Testing

### Testing Types:

* Functional Testing
* Model Validation
* UI Testing
* Edge Case Testing

### Sample Test Cases:

* Non-smoker with high BMI
* Young smoker
* Elderly non-smoker

---

# 🔹 Phase 10: Deployment (Optional)

Deployment Platforms:

* Streamlit Cloud
* Render
* Heroku
* AWS

---

# 📊 6. Technology Stack

| Component     | Technology          |
| ------------- | ------------------- |
| Language      | Python              |
| ML Library    | Scikit-learn        |
| Data Handling | Pandas, NumPy       |
| UI            | Streamlit           |
| Model Saving  | Pickle              |
| Development   | VS Code / Terminal  |

---

## 2. System Architecture
The SmartPolicy system follows a modular architecture designed for scalability and performance.

### 2.1. High-Level Architecture
```mermaid
graph TD
    User[User Interface (Streamlit)] -->|Input Data| Preprocessing[Data Preprocessing]
    Preprocessing -->|Feature Engineering| ML_Layer[Machine Learning Layer]
    ML_Layer -->|Regression| Charge_Model[XGBoost Regressor]
    ML_Layer -->|Classification| Risk_Model[Random Forest Classifier]
    Charge_Model -->|Prediction| Output[Results & PDF Report]
    Risk_Model -->|Risk Category| Output
```

### 2.2. Data Flow Diagram (DFD)
1.  **Input**: User enters personal, medical, and lifestyle data via the Web UI.
2.  **Processing**:
    *   Data is validated and converted to a DataFrame.
    *   **Feature Engineering**: BMI Category and Age Groups are derived.
    *   **Encoding**: Categorical variables are One-Hot Encoded.
    *   **Scaling**: Numerical variables are Standard Scaled.
3.  **Inference**:
    *   **Charge Prediction**: The XGBoost model estimates the premium.
    *   **Risk Assessment**: The Random Forest model classifies the user into Low, Medium, or High risk.
4.  **Output**: Predictions are displayed via Gauge Charts and a downloadable PDF report.

## 3. Algorithms and Mathematics
### 3.1. Gradient Boosting (XGBoost)
We utilize **Extreme Gradient Boosting (XGBoost)** for regression. It is an ensemble technique that builds models sequentially.
*   **Objective Function**: Minimize the Regularized Loss Function.
    $$ \mathcal{L}(\phi) = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k) $$
    Where:
    *   $l$ is the differentiable convex loss function (e.g., MSE).
    *   $\Omega$ is the regularization term to prevent overfitting.

### 3.2. Random Forest Classification
Used for Risk Categorization. It constructs a multitude of decision trees at training time.
*   **Gini Impurity** (Split Criterion):
    $$ Gini = 1 - \sum_{i=1}^{C} (p_i)^2 $$
    Where $p_i$ is the probability of an item being classified into specific class.

## 4. Project Results
*   **Model Accuracy**:
    *   **R² Score (Charges)**: > 0.85 (High Precision)
    *   **Risk Classification Accuracy**: > 95%
*   **Key Insights**:
    *   Smokers have significantly higher insurance costs.
    *   High BMI combined with smoking leads to exponential cost increases.
    *   Age is a linear factor, but chronic diseases introduce non-linear spikes.

## 5. Future Scope
*   **Deployment**: Deploying the app to AWS/Azure using Docker.
*   **Real-time Learning**: Implementing an online learning pipeline to update the model with new user data.
*   **IoT Integration**: Connecting with wearables (Fitbit/Apple Watch) to fetch real-time health vitals.

## 6. Conclusion
The SmartPolicy project successfully demonstrates the application of Advanced Machine Learning in the InsurTech domain. By integrating comprehensive health datasets and utilizing state-of-the-art algorithms, the system provides accurate, explainable, and user-friendly insurance cost estimations.
This project enhances understanding of data science workflow from preprocessing to deployment and reflects industry-level implementation standards.
