# Customer-churn-predictor
# Customer Churn Predictor 🔮

A machine learning web application that predicts customer churn for a telecommunications company. Built with XGBoost and deployed live on Streamlit Cloud. Containerised with Docker for reproducible local deployment.

**🌐 Live App:** [customer-churn-predictor on Streamlit Cloud](https://customer-churn-predictor-ap55uhv4scci2qsftjltu4.streamlit.app/)

---

## What This Project Does

Customer churn — when a customer stops using a service and is one of the most costly problems in subscription businesses. Acquiring a new customer costs 5–7x more than retaining an existing one. This app predicts the probability that a customer will churn based on their contract details, usage patterns, and service subscriptions.

You enter a customer's details into the app — contract type, tenure, monthly charges, which services they use — and the model instantly returns:
- **Churn probability** (0–100%)
- **Risk tier** (Low / Medium / High)
- **Actionable retention advice** based on the risk level

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC (test set) | **0.836** |
| Algorithm | XGBoost Classifier |
| Dataset | IBM Telco Customer Churn (7,043 customers) |
| Class imbalance handling | scale_pos_weight |

---

## Features Engineered

- `charges_per_tenure` — monthly charges divided by tenure months (cost efficiency signal)
- `total_services_count` — total number of active services (engagement signal)
- Standard encoded categorical features (contract type, payment method, internet service)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost, scikit-learn |
| Web App | Streamlit |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |
| Containerisation | Docker |
| Version Control | Git / GitHub |

---

## 🐳 Docker — Run Locally

Docker allows you to run this application on any machine without worrying about Python versions, dependencies, or environment setup. Everything the app needs is packaged inside the Docker container.

**What Docker does here:**
1. Pulls a lightweight Python 3.10 environment
2. Installs all required packages from `requirements.txt`
3. Copies the app files into the container
4. Starts the Streamlit server on port 8501
5. The app runs identically on any machine — Windows, Mac, or Linux

**Prerequisites:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Build the image:**
```bash
docker build -t churn-predictor .
```

**Run the container:**
```bash
docker run -p 8501:8501 churn-predictor
```

**Open in browser:**
```
http://localhost:8501
```

**Stop the container:**
```bash
docker stop $(docker ps -q --filter ancestor=churn-predictor)
```

---

## Project Structure

```
Customer-churn-predictor/
│
├── app.py                          # Streamlit web application
├── churn_model.pkl                 # Trained XGBoost model
├── churn_processed.csv             # Processed dataset
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker containerisation
├── fig1_churn_eda.png              # EDA visualisation
├── fig2_feature_importance.png     # Feature importance plot
├── fig3_Tenure_Distribution_by_Churn.png  # Tenure analysis
└── README.md
```

---

## How to Run Locally (without Docker)

```bash
# Clone the repository
git clone https://github.com/Fakiha1407/Customer-churn-predictor.git
cd Customer-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```


## Key Findings

- Customers on **month-to-month contracts** churn at significantly higher rates than annual/two-year contracts
- **High monthly charges** combined with **short tenure** is the strongest churn signal
- Customers using **fibre optic internet** show higher churn than DSL customers
- Fraud occurs exclusively in customers with **electronic check** payment method


## Author

**Fakiha Balouch**  
M.Sc. Data Science · FAU Erlangen-Nürnberg  
[LinkedIn](https://linkedin.com/in/FakihaBalouch) · [GitHub](https://github.com/Fakiha1407)
