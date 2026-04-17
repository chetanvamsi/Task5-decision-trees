import pandas as pd

data = pd.read_csv("dataset.csv")
from sklearn.model_selection import train_test_split

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split done")

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Model trained")
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

model_limited = DecisionTreeClassifier(max_depth=3)
model_limited.fit(X_train, y_train)

y_pred2 = model_limited.predict(X_test)

print("Controlled Accuracy:", accuracy_score(y_test, y_pred2))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(rf.feature_importances_, index=X.columns)

print(importance)

importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Average:", scores.mean())