import pandas as pd
data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQZ0yYIzNdSb16GTMcbE3wgVTYF4uLYDM6aOL2ZWXqkpInJ4oYG2wo5fwZ56ivXi4ya5aGuOjOs_oJ9/pub?output=csv")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Keep only temperature, humidity, and oxygen columns
X = data[['Temperature', 'Humidity', 'Oxygen']]
y = data['Fire Occurrence']

# Normalize numerical columns
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

y_pred = logreg.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test,y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
new_data = pd.DataFrame({
    'Temperature': [10], 
    'Humidity': [70], 
    'Oxygen': [30]
})

# Preprocess new data
new_data = scaler.transform(new_data)

# Make predictions
new_pred = logreg.predict(new_data)

print(new_pred)
