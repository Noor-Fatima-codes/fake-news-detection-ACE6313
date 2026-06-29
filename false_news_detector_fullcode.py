# Fake News Detection Project (SDG 16)
# Phases 1–4 with progress messages

import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Download NLTK resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

print("🚀 Starting Fake News Detection Pipeline...")

# -------------------------------
# Phase 1: Load & Explore Dataset
# -------------------------------
print("\n📊 Phase 1: Loading and exploring dataset...")

df_fake = pd.read_csv("fake.csv")
df_true = pd.read_csv("true.csv")

df_fake["label"] = 1
df_true["label"] = 0
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.drop_duplicates(inplace=True)
df['subject'] = df['subject'].str.lower().str.strip()

print("Class balance:\n", df['label'].value_counts())
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
print("Average word count:", df['word_count'].mean())
print("✅ Phase 1 complete.")

# -------------------------------
# Phase 2: Preprocessing Pipeline
# -------------------------------
print("\n🧹 Phase 2: Preprocessing text...")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_text'])

svd = TruncatedSVD(n_components=300)
X_reduced = svd.fit_transform(X_tfidf)

X = X_reduced
y = df['label'].values

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("✅ Phase 2 complete.")

# -------------------------------
# Phase 3: Model Training
# -------------------------------
print("\n🤖 Phase 3: Training models...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T
print("\nModel Results:\n", results_df)

best_model_name = results_df['F1'].idxmax()
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

print(f"✅ Phase 3 complete. Best model so far: {best_model_name}")

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# Hyperparameter tuning example
print("\n Running hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters (Random Forest):", grid.best_params_)
print("Best F1 score (Random Forest):", grid.best_score_)


# Hyperparameter tuning for svm

print("\nRunning hyperparameter tuning for SVM...")

svm_param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=3, scoring='f1', n_jobs=-1)
svm_grid.fit(X_train, y_train)

print("Best SVM parameters:", svm_grid.best_params_)
print("Best F1 score (SVM):", svm_grid.best_score_)

# -------------------------------
# Phase 4: Report Prep
# -------------------------------
print("\n📝 Phase 4: Saving results...")

results_df.to_csv("model_results.csv", index=True)

tuning_results = pd.DataFrame({
    "Model": ["Random Forest", "SVM"],
    "Best Parameters": [grid.best_params_, svm_grid.best_params_],
    "Best F1 Score": [grid.best_score_, svm_grid.best_score_],
})
tuning_results.to_csv("tuning_results.csv", index=False)

joblib.dump(best_model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(svd, "svd_transformer.pkl")
joblib.dump(best_model_name, "best_model_name.pkl")

print("Results saved to model_results.csv")

print("Model saved to fake_news_model.pkl")
print("TF-IDF saved to tfidf_vectorizer.pkl")
print("SVD transformer saved to svd_transformer.pkl")
print("Best model saved to model_results.csv")


print("\n🎯 Summary:")
print("Best model:", best_model_name)
print("Pipeline complete. All outputs generated.")
