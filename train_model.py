### import all the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for saving
import os      # Import os for creating directories

print("Starting model training process...")

# --- 1. Load and Prepare Data ---
df = pd.read_csv("model/dataset/spam.csv")
data = df.where((pd.notnull(df)), "")
data.loc[data["Category"] == "ham", "Category"] = 0
data.loc[data["Category"] == "spam", "Category"] = 1

X = data["Message"]
Y = data["Category"].astype('int')

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# --- 2. Create and Fit the Vectorizer ---
# This is the vectorizer object we need to save
# Its variable name is 'feature_extraction'
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# --- 3. Train the Model ---
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# --- 4. Evaluate the Model (Optional but good practice) ---
predictions = model.predict(X_test_features)
accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy:.6f}")

# --- 5. Save the Model and the Vectorizer ---
output_dir = 'model'
os.makedirs(output_dir, exist_ok=True)

# Save the trained model
joblib.dump(model, os.path.join(output_dir, 'spam_detector_model.joblib'))

# Save the fitted vectorizer
joblib.dump(feature_extraction, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))

print(f"Model and vectorizer have been saved to the '{output_dir}' directory.")
print("Training complete!")