import pandas as pd
import numpy as np
import spacy
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from scipy.sparse import hstack, csr_matrix

"""
WARNING: This file takes a while to complete
"""

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv('gender-classifier-DFE-791531.csv', encoding='ISO-8859-1')

# Filter out brands and profiles with missing gender
df = df[df['gender'].isin(['male', 'female'])]
df = df.dropna(subset=['gender'])
df = df.reset_index(drop=True)

# spaCy tokenizer wrapper
def spacy_tokenizer(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Prepare target variable
y = df['gender'].map({'male': 0, 'female': 1}).values  # 0: male, 1: female

# Caching tfidf vectors to avoid recomputing
def get_or_cache_tfidf(field, vectorizer, filename):
    try:
        # Load both the TF-IDF matrix and the vectorizer
        X_cached = joblib.load(filename)
        vectorizer = joblib.load(filename.replace(".pkl","_vectorizer.pkl"))
        print(f"Loaded cached TF-IDF for {field}")
    except FileNotFoundError:
        X_cached = vectorizer.fit_transform(df[field].fillna(''))
        #save both tfidf matrix and vectorizer.
        joblib.dump(vectorizer,filename.replace(".pkl","_vectorizer.pkl"))
        joblib.dump(X_cached, filename)

        print(f"Computed and cached TF-IDF for {field}")
    return X_cached, vectorizer

# TF-IDF vectorizers
tfidf_text = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
tfidf_desc = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
tfidf_name = TfidfVectorizer(ngram_range=(1, 1)) # removed tokenizer because it was not used for name.

# Get TF-IDF matrices
X_text, tfidf_text = get_or_cache_tfidf('text', tfidf_text, 'tfidf_text.pkl')
X_desc, tfidf_desc = get_or_cache_tfidf('description', tfidf_desc, 'tfidf_desc.pkl')
X_name, tfidf_name = get_or_cache_tfidf('name', tfidf_name, 'tfidf_name.pkl')

# Numerical features
df['tweet_count'] = df['tweet_count'].fillna(0)
df['retweet_count'] = df['retweet_count'].fillna(0)
df['fav_number'] = df['fav_number'].fillna(0)

# Scale `tweet_count` (dense)
scaler_tweet = StandardScaler()
tweet_count_scaled = scaler_tweet.fit_transform(df[['tweet_count']].values)

# Scale `retweet_count` and `fav_number` (sparse)
scaler_other = StandardScaler()
other_features_scaled = scaler_other.fit_transform(df[['retweet_count', 'fav_number']].values)
other_sparse = csr_matrix(other_features_scaled)

# Combine all features
X = hstack([X_desc,X_name,X_text,other_sparse])
# retweet_count & fav_number as sparse
# tweet_count stays dense and will be handled separately during training

# Classifier
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
}

# k-Fold cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n===== {model_name} =====")
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        tweet_train = tweet_count_scaled[train_idx]
        tweet_test = tweet_count_scaled[test_idx]

        X_train_full = hstack([X_train,tweet_train])
        X_test_full = hstack([X_test,tweet_test])

        model.fit(X_train_full, y_train)
        y_pred = model.predict(X_test_full)
        y_prob = model.predict_proba(X_test_full)[:, 1]  # For ROC-AUC

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_prob))

    print("Accuracy per fold:", accuracy_scores)
    print("Precision per fold:", precision_scores)
    print("Recall per fold:", recall_scores)
    print("F1 per fold:", f1_scores)
    print("ROC-AUC per fold:", roc_auc_scores)
    print("Average Accuracy:", np.mean(accuracy_scores))
    print("Average Precision:", np.mean(precision_scores))
    print("Average Recall:", np.mean(recall_scores))
    print("Average F1:", np.mean(f1_scores))
    print("Average ROC-AUC:", np.mean(roc_auc_scores))

    # Confusion matrix and classification report on the last fold for illustration
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, 'PreviousVecotrizations/gender_classifier_model.pkl')
    joblib.dump(scaler_other, 'PreviousVecotrizations/scaler.pkl')
    joblib.dump(scaler_tweet, 'PreviousVecotrizations/scaler_tweet.pkl')

# ========== GRAPHING Section ==========

folds = list(range(1, k + 1))

# Plot each metric
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(folds, accuracy_scores, marker='o', label='Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy per Fold')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(folds, precision_scores, marker='o', label='Precision', color='green')
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.title('Precision per Fold')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(folds, recall_scores, marker='o', label='Recall', color='red')
plt.xlabel('Fold')
plt.ylabel('Recall')
plt.title('Recall per Fold')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(folds, f1_scores, marker='o', label='F1-Score', color='purple')
plt.xlabel('Fold')
plt.ylabel('F1-Score')
plt.title('F1-Score per Fold')
plt.grid(True)

plt.tight_layout()
plt.show()

# ROC-AUC separately
plt.figure(figsize=(7, 5))
plt.plot(folds, roc_auc_scores, marker='o', label='ROC-AUC', color='orange')
plt.xlabel('Fold')
plt.ylabel('ROC-AUC')
plt.title('ROC-AUC per Fold')
plt.grid(True)
plt.legend()
plt.show()

# ========== TESTING Section ==========

# Load combined CSV
df_test = pd.read_csv('Assignment1Combined.csv')

# True gender labels
df_test = df_test[df_test['gender'].isin(['male', 'female'])]
df_test['true_gender']= df_test['gender'].map({'male': 0, 'female': 1})

# Load fitted objects
model = joblib.load('PreviousVecotrizations/gender_classifier_model.pkl')
tfidf_name_vectorizer = joblib.load('tfidf_name_vectorizer.pkl')  # Load the vectorizer
tfidf_text_vectorizer = joblib.load('tfidf_text_vectorizer.pkl')  # Load the vectorizer
tfidf_desc_vectorizer = joblib.load('tfidf_desc_vectorizer.pkl')  # Load the vectorizer
scaler = joblib.load('PreviousVecotrizations/scaler.pkl')
scaler_tweet = joblib.load('PreviousVecotrizations/scaler_tweet.pkl')

# Transform test data
X_name_test = tfidf_name_vectorizer.transform(df_test['name'].fillna(''))
X_text_test = tfidf_text_vectorizer.transform(df_test['text'].fillna(''))

# Handle missing 'tweet_count'
if 'tweet_count' in df_test.columns:
    tweet_count_test = df_test['tweet_count'].fillna(0).values.reshape(-1, 1)
    tweet_count_test_scaled = scaler_tweet.transform(tweet_count_test)
    tweet_count_test_sparse = csr_matrix(tweet_count_test_scaled)
else:
    print("Warning: 'tweet_count' column not found in test data. Using zeros.")
    tweet_count_test_scaled = np.zeros((len(df_test), 1))
    tweet_count_test_sparse = csr_matrix(tweet_count_test_scaled)

# Handle missing 'retweet_count' and 'fav_number'
if all(col in df_test.columns for col in ['retweet_count', 'fav_number']):
    dummy_numerical = df_test[['retweet_count', 'fav_number']].fillna(0).values
    dummy_numerical_scaled = scaler.transform(dummy_numerical)
    dummy_numerical_sparse = csr_matrix(dummy_numerical_scaled)
else:
    print("Warning: 'retweet_count' or 'fav_number' columns not found in test data. Using zeros.")
    dummy_numerical_scaled = np.zeros((len(df_test), 2))
    dummy_numerical_sparse = csr_matrix(dummy_numerical_scaled)

# Handle missing 'description'
if 'description' in df_test.columns:
    X_desc_test = tfidf_desc_vectorizer.transform(df_test['description'].fillna(''))
else:
    print("Warning: 'description' column not found in test data. Using zeros.")
    X_desc_test = csr_matrix(np.zeros((len(df_test), tfidf_desc_vectorizer.vocabulary_.__len__())))  # Create a sparse matrix of zeros

# Get the number of features from the training data
num_features_train = X.shape[1]

# Combine all test features
X_test_Assign1_full = hstack([X_desc_test, X_name_test, X_text_test, dummy_numerical_sparse, tweet_count_test_sparse])

# Predict
y_pred = model.predict(X_test_Assign1_full)

# Add predictions to dataframe
df_test['predicted_gender'] = y_pred
df_test['predicted_gender'] = df_test['predicted_gender'].map({0: 'male', 1: 'female'})

print(df_test[['name', 'text', 'gender', 'predicted_gender']])

# Evaluation
print("\nClassification Report on Assignment1Combined.csv:")
print(classification_report(df_test['true_gender'], y_pred))

# ========== Additional Graphing Section ==========

# Get evaluation metrics
accuracy = accuracy_score(df_test['true_gender'], y_pred)
precision = precision_score(df_test['true_gender'], y_pred)
recall = recall_score(df_test['true_gender'], y_pred)
f1 = f1_score(df_test['true_gender'], y_pred)
roc_auc = roc_auc_score(df_test['true_gender'], model.predict_proba(X_test_Assign1_full)[:, 1])

# Create a bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
print(f"The accuracy on assignment 1 is {accuracy}, precision is {precision}, recall is {recall}, f1 is {f1}, Roc-AUC is {roc_auc}")

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.title('Evaluation Metrics on Assignment1Combined.csv')
plt.ylabel('Value')
plt.show()