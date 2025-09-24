import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from xgboost import XGBClassifier

df = pd.read_csv("KaggleV2-May-2016.csv")

# Data cleanup
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'F':1,'M':0})
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Remove entries with negative waiting times or unrealistic age
df = df[(df['WaitDays'] >= 0) & (df['Age'] >= 0) & (df['Age'] <= 100)]

# Select features
features = ["Age", "Gender", "Scholarship", "SMS_received", "WaitDays",
    "Hipertension", "Diabetes", "Alcoholism", "Handcap"]

X = df[features]
y = df['No-show']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle class imbalance
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "no_show_model.pkl")
print("Model saved as no_show_model.pkl")


