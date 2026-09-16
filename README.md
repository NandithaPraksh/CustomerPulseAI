# CustomerPulse AI

### Automated Customer Churn Prediction & Retention Analytics Platform

CustomerPulse AI is a Streamlit-based machine learning application that helps businesses understand customer behavior and identify customers who may be at risk of churn.

Instead of requiring users to manually clean data, perform exploratory analysis, engineer features, and train machine learning models, CustomerPulse AI automates the complete workflow from **CSV upload to customer risk analysis**.

---

## What Does CustomerPulse AI Do?

A business can upload its customer transaction data, and the application automatically:

1. Validates and cleans the uploaded dataset
2. Removes or handles invalid transaction records
3. Processes customer purchase history
4. Creates customer-level behavioral features
5. Defines a temporal churn label
6. Performs exploratory data analysis
7. Trains multiple machine learning models
8. Compares model performance
9. Predicts individual customer churn probability
10. Classifies customers into risk levels
11. Provides model-based explanations for customer risk
12. Generates actionable business insights

### Workflow

```text
CSV Upload
    ↓
Data Validation & Cleaning
    ↓
Transaction Processing
    ↓
Customer Feature Engineering
    ↓
Churn Label Creation
    ↓
Exploratory Data Analysis
    ↓
Machine Learning Models
    ↓
Model Evaluation & Comparison
    ↓
Churn Probability Prediction
    ↓
Customer Risk Classification
    ↓
Explainability
    ↓
Business Insights
```

---

## Dataset

The project is developed using the **UCI Online Retail dataset**.

The dataset contains transaction-level information including:

* Invoice number
* Product/stock code
* Product description
* Quantity
* Invoice date
* Unit price
* Customer ID
* Country

The application transforms these transaction-level records into customer-level behavioral features suitable for churn prediction.

### Important

The dataset is intentionally **not included in the GitHub repository**.

Users can upload the CSV directly through the application's **Upload & Data Quality** page.

---

## Automated Data Processing

CustomerPulse AI automatically handles several data-quality issues before modeling, including:

* Missing customer identifiers
* Duplicate transactions
* Cancelled invoices
* Returns and negative quantities
* Invalid or zero-price transactions
* Date conversion
* Monetary value calculation
* Transaction filtering
* Customer-level aggregation

This allows the application to work as an automated analytics pipeline rather than requiring manual preprocessing.

---

##  Customer Feature Engineering

The application converts transaction history into customer-level behavioral features such as:

* **Recency** – How recently the customer purchased
* **Frequency** – Number of qualifying purchases
* **Monetary Value** – Customer spending
* **Average Order Value**
* **Total Items Purchased**
* **Distinct Products**
* **Distinct Invoices**
* **Return Rate**
* **Cancelled Amount**
* **Customer Tenure**
* **Inter-Purchase Gap**
* **Purchase Span**
* **Purchases in Recent Period**
* **Revenue in Recent Period**
* **Number of Returns**
* **RFM-related features**

These features allow the models to learn behavioral patterns associated with customer retention and churn.

---

##  Temporal Churn Definition

CustomerPulse AI uses a time-based churn definition instead of randomly assigning churn labels.

For the development dataset:

**Snapshot date:** September 30, 2011

Customer behavior before the snapshot is used for feature engineering.

The subsequent observation period is used to determine whether the customer returned.

```text
Historical Customer Activity
          ↓
    September 30, 2011
          ↓
   Observation Period
          ↓
Returned? ──────── No → Churn
    │
    └────────────── Yes → Retained
```

This approach helps prevent temporal data leakage between the features and the churn target.

> Note: The available dataset ends on December 9, 2011, so the observation period is shorter than a full 90-day period.

---

## 📈 Exploratory Data Analysis

The application automatically generates visualizations to understand customer and transaction behavior.

Examples include:

* Churn distribution
* Recency distribution
* Frequency distribution
* Monetary value distribution
* Customer behavior by churn status
* Correlation heatmap
* Monthly transaction trends
* Monthly revenue trends
* Country-level analysis
* Return behavior
* RFM-related analysis

---

## 🤖 Machine Learning

CustomerPulse AI trains and compares multiple classification models.

### Models

* Logistic Regression
* Random Forest
* HistGradientBoosting
* K-Nearest Neighbors

Models are evaluated using stratified cross-validation and multiple classification metrics.

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

The application also provides:

* Model comparison
* Confusion matrices
* ROC curves
* Precision-Recall curves

---

##  Customer Risk Prediction

For each customer, the application generates a **churn probability**.

Customers are grouped into three risk categories:

| Risk Level | Churn Probability |
| ---------- | ----------------: |
| Low        |            < 0.33 |
| Medium     |     0.33 – < 0.66 |
| High       |            ≥ 0.66 |

This allows businesses to identify customers who may require retention attention.

---

##  Customer Risk Explorer

The Customer Risk Explorer allows users to investigate individual customers.

For a selected customer, the application can display:

* Customer features
* Churn probability
* Risk category
* Behavioral information
* Model-derived factors contributing to the prediction

Where supported, the application uses **SHAP-based explainability**. A model-specific fallback explanation is used when SHAP is not available for a particular model.

---

##  Business Insights

CustomerPulse AI converts analytical and model outputs into business-oriented insights.

The application can highlight:

* Customer retention patterns
* High-risk customer segments
* Recent purchasing behavior
* Spending patterns
* Return behavior
* Geographic patterns
* Factors associated with predicted churn

The goal is to move beyond simply predicting churn and provide information that can support customer-retention strategies.

---

##  Application Pages

The application contains the following sections:

###  Home

Introduction to CustomerPulse AI and the overall workflow.

###  Upload & Data Quality

Upload customer transaction data and review automated data validation and cleaning.

###  Exploratory Data Analysis

Explore customer behavior and transaction patterns through automatically generated visualizations.

###  Model Performance

Compare machine learning models and examine evaluation metrics and diagnostic plots.

###  Churn Predictions

View customer-level churn probabilities and risk classifications.

###  Customer Risk Explorer

Investigate individual customers and understand model-derived risk factors.

###  Business Insights

View summarized analytical findings and customer-retention insights.

---

## Technology Stack

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Application & ML development |
| Streamlit    | Interactive web application  |
| Pandas       | Data manipulation            |
| NumPy        | Numerical computing          |
| Matplotlib   | Visualization                |
| Seaborn      | Statistical visualization    |
| Scikit-learn | Machine learning             |
| SHAP         | Model explainability         |

---

## Project Structure

```text
CustomerPulseAI/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENT.md
└── data.csv
```

`data.csv` is used as the development dataset and is excluded from version control through `.gitignore`.

---

##  Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/NandithaPraksh/CustomerPulseAI.git
cd CustomerPulseAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

##  Deployment

CustomerPulse AI can be deployed using **Streamlit Community Cloud**.

Deployment configuration:

```text
Repository: CustomerPulseAI
Branch: main
Main file: app.py
```

After deployment, users can upload their own compatible CSV dataset through the application.

---

##  Data & Privacy

Customer data should be handled responsibly.

The application is designed around user-uploaded datasets and does not require a separate database or backend server.

Do not upload confidential or personally identifiable customer information to a public deployment unless appropriate privacy and security controls are in place.

---

##  Project Objective

CustomerPulse AI demonstrates how machine learning can be integrated into an end-to-end customer analytics workflow.

The project focuses on:

* Automated data preparation
* Customer behavior analysis
* Temporal churn prediction
* Machine learning model comparison
* Customer-level risk identification
* Model explainability
* Business-oriented analytics

Rather than treating churn prediction as only a machine learning classification problem, CustomerPulse AI combines **data analytics, machine learning, visualization, explainability, and business insights** into a single application.

---

##  Author

**Nanditha Praksh**

Computer Science & Engineering

GitHub: `https://github.com/NandithaPraksh`

---

##  Future Enhancements

Potential future improvements include:

* Automated model retraining
* Support for additional customer datasets
* Advanced hyperparameter optimization
* More explainability techniques
* Customer segmentation
* Retention campaign recommendations
* Real-time customer monitoring
* Database integration
* Automated email/notification workflows
