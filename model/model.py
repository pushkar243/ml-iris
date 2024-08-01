import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load dataset
data = pd.read_csv('../data/iris.csv')


# Preprocess data
X = data.drop('species', axis=1)
y = data['species']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=42)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Save model
joblib.dump(model, 'model.pkl')
