import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("gesture_data.csv", header=None)

# First column = label, rest = features
X = df.iloc[:, 1:]  
y = df.iloc[:, 0]   

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save trained model
pickle.dump(model, open("gesture_model.pkl", "wb"))
print("Model saved as gesture_model.pkl")