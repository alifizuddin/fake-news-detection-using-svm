import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # Make text lowercase
    text = re.sub(r"http\S+|[^a-z\s]", "", text) # Remove links and punctuation
    text = " ".join(word for word in text.split() if word not in stop_words) # Remove stopwords like "the", "is", "in"
    return text

df = pd.read_csv("dataset/FakeNewsNet.csv")
df = df[['title', 'real']]      
df.columns = ['text', 'label']   
df['text'] = df['text'].apply(clean_text)  # Clean text

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english' ) # Turn all the cleaned text into a matrix of numbers
X = vectorizer.fit_transform(df['text'])  # All input text converted to numbers
y = df['label']                           # Target values (0 = fake, 1 = real)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Print the results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

def predict_news(title):
    title = clean_text(title)
    title_vec = vectorizer.transform([title])
    result = model.predict(title_vec)
    return "REAL" if result[0] == 1 else "FAKE"

def run_gui():
    window = tk.Tk()
    window.title("Fake News Detector")

    tk.Label(window, text="Enter News Title:", font=('Arial', 12)).pack(pady=10)
    entry = tk.Entry(window, width=60)
    entry.pack(pady=5)

    def on_check():
        title = entry.get()
        if not title:
            messagebox.showwarning("Input Error", "Please enter a news title.")
            return
        result = predict_news(title)
        messagebox.showinfo("Prediction Result", f"This news is likely: {result}")

    tk.Button(window, text="Check", command=on_check).pack(pady=10)
    window.mainloop()

run_gui()
