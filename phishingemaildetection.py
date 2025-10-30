# 🔐 Phishing Email Detection Project
# By Ayush Agarwal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset (You can replace this with your own CSV file)
# Sample dataset columns: ['EmailText', 'Label']
# Label: 1 = Phishing, 0 = Legitimate
data = {
    'EmailText': [
        'Congratulations! You have won a $1000 gift card. Click here to claim.',
        'Your Amazon order has been shipped.',
        'Update your bank details immediately to avoid suspension.',
        'Meeting rescheduled for 3 PM tomorrow.',
        'Verify your account to prevent deactivation.',
        'Lunch at 1 PM today?'
    ],
    'Label': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# 2. Data preprocessing
X = df['EmailText']
y = df['Label']

# 3. Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_features = vectorizer.fit_transform(X)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# 5. Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# 7. Test with custom email
def predict_email(email):
    input_features = vectorizer.transform([email])
    prediction = model.predict(input_features)
    return "⚠️ Phishing Email" if prediction[0] == 1 else "✅ Legitimate Email"

# Example
email_text = input("\nEnter an email text to analyze: ")
print("Result:", predict_email(email_text))
