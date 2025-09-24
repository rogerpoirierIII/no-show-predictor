import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import csv
from git.objects.submodule.util import sm_section
from narwhals.selectors import datetime
from pandas.io.common import file_exists
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from train_model import X_test, y_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("no_show_model.pkl")

# Navagation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["No-Show Predictor", "Data Visualization","Monitoring","Model Performance"])


if page == "No-Show Predictor":
    st.title("Patient No-Show Prediction")

    # Create Input Fields
    age = st.slider("Patient Age", 0, 100,50)
    gender = st.radio('Gender',['Male','Female'])
    scholarship = st.radio("Patient has a scholarship?",["Yes","No"])
    sms_received = st.radio("Enrolled in SMS Notifications?", ["Yes", "No"])
    st.text("Health Conditions")
    col1, col2, col3 = st.columns(3)

    with col1:
        hypertension = st.checkbox("Hypertension")

    with col2:
        diabetes = st.checkbox("Diabetes")

    with col3:
        alcoholism = st.checkbox("Alcoholism")

    handicap = st.selectbox('Handicap Level', [0,1,2,3,4])

    # date collection and conversion to WaitDays
    scheduled_date = st.date_input("Date Appointment was Scheduled")
    appointment_date = st.date_input("Appointment Date")

    if appointment_date and scheduled_date:
        wait_days = (appointment_date - scheduled_date).days
    else:
        wait_days = None

    # WaitDays Validation
    if wait_days < 0:
        st.error("Invalid input: Appointment date cannot be before the date it was scheduled!")

    # Convert inputs to DataFrame
    if st.button("Predict No-Show Risk"):
        input_data = pd.DataFrame([[
            age,
            1 if gender == 'Female' else 0,
            1 if scholarship == "Yes" else 0,
            1 if sms_received == "Yes" else 0,
            wait_days,
            1 if hypertension else 0,
            1 if diabetes else 0,
            1 if alcoholism else 0,
            handicap

            ]], columns=["Age", "Gender", "Scholarship", "SMS_received", "WaitDays",
            "Hipertension", "Diabetes", "Alcoholism", "Handcap"])

        # Prediction probability
        p = model.predict_proba(input_data)[0]
        no_show_probability = p[1]
        show_probability = p[0]

        #Threshold: if no-show probability is >= 50%
        prediction = 1 if no_show_probability >= 0.5 else 0

        # Predict button to display probability
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error(f"High Risk: Patient has a {no_show_probability *100:.1f}% chance of missing the next appointment.")
        else:
            st.success(f"Low Risk: Patient has a {(1 - show_probability)*100:.1f}% chance of missing the next appointment.")

    # Logging
        file_exists = os.path.isfile("prediction_log.csv")
        with open("prediction_log.csv", mode='a', newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Age','Gender','Scholarship','SMS_received','Wait Days','Hypertension','Diabetes','Alcoholism','Handicap', 'No-Show Probability',"Date Generated"])
            writer.writerow([age, gender, scholarship, sms_received, wait_days,
            hypertension, diabetes, alcoholism, handicap,
            round(no_show_probability, 4),
            datetime.todgay().strftime("%m/%d/%Y")])

elif page == "Data Visualization":
    st.title("Data Visualization")
# Load raw dataset (optional: in a sidebar toggle)
    df = pd.read_csv("KaggleV2-May-2016.csv")
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'F': 'Female', 'M': 'Male'})

# Visualization: No-show counts
    st.subheader("No-show Counts")
    st.bar_chart(df['No-show'].value_counts())

    # Visualization: No-show by age group
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 18, 35, 50, 65, 100], labels=["0–17", "18–34", "35–49", "50–64", "65+"])
    age_group_noshow = df.groupby("AgeGroup")["No-show"].mean()*100

    st.subheader("No-Show Rate by Age Group (%)")
    st.bar_chart(age_group_noshow)

    # Visualization: Gender pie chart
    st.subheader("Gender Distribution")
    gender_counts = df['Gender'].value_counts()
    st.write(gender_counts)
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

# Monitoring Dashboard
if page == "Monitoring":
    st.title("Monitoring Dashboard")
    if os.path.exists("prediction_log.csv"):
        log_df = pd.read_csv("prediction_log.csv")
        st.metric("Total Predictions", len(log_df))
        st.metric("Avg. No-Show Risk", f"{log_df['No-Show Probability'].mean():.2%}")

        st.subheader("Recent Predictions")
        st.dataframe(log_df.tail(10))
    else:
        st.warning("No predictions have been logged yet.")

# Performance Dashboard
if page == "Model Performance":
    st.title("Performance Dashboard")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    # Plot
    st.subheader("Model Performance Metrics")
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color="blue")
    ax.set_ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    st.pyplot(fig)

    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Show", "No-show"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
