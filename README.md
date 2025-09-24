# Medical Appointment No-Show Prediction

Predict whether a patient will miss a scheduled medical appointment using machine learning.  
This project demonstrates **end-to-end data science** and **full-stack deployment** with a real-world healthcare use case.

---

## Project Overview
Missed medical appointments (“no-shows”) create costly scheduling gaps, reduce provider productivity, and delay patient care.  
This application uses a machine-learning model trained on the public [Kaggle Medical Appointment No Shows dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments) to predict the likelihood that a patient will not attend an appointment.

**Key Features**
- **Interactive Streamlit Web App** – User-friendly dashboard for entering patient data and receiving instant risk predictions.
- **XGBoost Model** – Gradient-boosted decision trees tuned for class imbalance.
- **Data Exploration & Visualizations** – Built-in charts for dataset insights, feature importance, and model evaluation.
- **Monitoring & Logging** – Automatic logging of predictions with timestamps for performance tracking.

---

## Tech Stack
- **Language:** Python 3.11  
- **Frameworks & Libraries:** Streamlit, XGBoost, scikit-learn, pandas, matplotlib, joblib  
- **Environment:** Windows 10/11, cross-platform compatible  
- **Deployment:** Local or cloud (Streamlit Community Cloud ready)

---

## Quick Start

### Prerequisites
- [Python 3.11+](https://www.python.org/downloads/)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/no-show-predictor.git
cd no-show-predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
