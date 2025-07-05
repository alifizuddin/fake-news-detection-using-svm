import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # Make text lowercase
    text = re.sub(r"http\S+|[^a-z\s]", "", text) # Remove links and punctuation
    text = " ".join(word for word in text.split() if word not in stop_words) # Remove stopwords like "the", "is", "in"
    return text

df = pd.read_csv("data/FakeNewsNet.csv")
df = df[['title', 'real']]      
df.columns = ['text', 'label']   
df['text'] = df['text'].apply(clean_text)  # Clean text

vectorizer = TfidfVectorizer(max_features=5000) # Turn all the cleaned text into a matrix of numbers
X = vectorizer.fit_transform(df['text'])  # All input text converted to numbers
y = df['label']                           # Target values (0 = fake, 1 = real)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearSVC()
model.fit(X_train, y_train)

# Step 5: Test the model
y_pred = model.predict(X_test)

# Print the results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
