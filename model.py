import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv("placementdata.csv")

le = LabelEncoder()

data["ExtracurricularActivities"] = le.fit_transform(
    data["ExtracurricularActivities"]
)

data["PlacementTraining"] = le.fit_transform(
    data["PlacementTraining"]
)

data["PlacementStatus"] = le.fit_transform(
    data["PlacementStatus"]
)

X = data.drop(["StudentID", "PlacementStatus"], axis=1)
y = data["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42 )

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print(importance.sort_values(by='Importance', ascending=False))
import pickle

pickle.dump(model, open("placement_model.pkl", "wb"))

print("Model Saved Successfully")