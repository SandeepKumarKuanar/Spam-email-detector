####### this is the static file that was first created to see if the model is working or not, before creating the dynamic version of it in 'app.py' using DJango as the backend layer and the frontend #######
### import all the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("model/dataset/spam.csv")
data = df.where((pd.notnull(df)), "")
# print(data.head())
# print(data.shape)
## Renaming the columns
data.loc[
    data["Category"] == "ham",
    "Category",
] = 0
data.loc[
    data["Category"] == "spam",
    "Category",
] = 1
# print(data.head())
## separating the data and label
X = data["Message"]
Y = data["Category"]
# print(Y)

## Splitting the data into training data and testing data
# 0.2 == 20% of the data will be used for testing, and 80% for training
# random_state is used to ensure that the results are reproducible and consistent across different runs
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
# print(X.shape, X_train.shape, X_test.shape)

### Transform the text data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
# print(X_train_features, X_test_features)

## converting Y_train and Y_test values as integers
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

### Training the model
model = LogisticRegression()
## with training data
model.fit(X_train_features, Y_train)
predictions_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, predictions_on_training_data)
print("Accuracy on training data : ", accuracy_on_training_data)

## with testing data
predictions_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, predictions_on_test_data)
print("Accuracy on test data : ", accuracy_on_test_data)
# print(predictions_on_training_data)

#### Making a predictive system
User_input_mail = input("Enter the email:\n")
input_data_features = feature_extraction.transform([User_input_mail])
prediction = model.predict(input_data_features)
# extra logic
if prediction[0] == 1:
    print("\nIt's a spam mail")
else:
    print("\nIt's a ham mail")

user_correct = input("Was the prediction correct? (yes/no): ").strip().lower()
final_label_to_save = None
# --- STEP 3: Determine the correct final label ---
if user_correct == "yes":
    # The model was right. Use its original prediction.
    final_label_to_save = prediction[0]
    print("Thank you for confirming!")
elif user_correct == "no":
    # The model was wrong. Ask the user for the true label.
    correct_label_str = (
        input("Please provide the correct label (spam/ham): ").strip().lower()
    )
    if correct_label_str == "spam":
        final_label_to_save = 1
    elif correct_label_str == "ham":
        final_label_to_save = 0
    else:
        print("Invalid label provided.")
else:
    print("Invalid input. Please enter 'yes' or 'no'.")
# --- STEP 4: Save the final, verified data to the file ---
if final_label_to_save is not None:
    email_data = {"Category": [final_label_to_save], "Message": [User_input_mail]}
    email_df = pd.DataFrame(email_data)
    email_df.to_csv("model/dataset/spam.csv", mode="a", index=False, header=False)
    print("The dataset has been updated.")
