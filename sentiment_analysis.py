import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sample_data import create_sample_data
import joblib

# Load data
data = create_sample_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2, random_state=42, stratify=data['sentiment']
)

# Create a pipeline for SVC
svc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', SVC())
])

# Define hyperparameters to search for SVC
svc_param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['rbf', 'linear']
}

# Perform grid search with cross-validation for SVC
svc_grid_search = GridSearchCV(svc_pipeline, svc_param_grid, cv=5, n_jobs=-1)
svc_grid_search.fit(X_train, y_train)

# Save the best SVC model
joblib.dump(svc_grid_search.best_estimator_, 'best_svc_model.pkl')

# Print best parameters for SVC
print("Best SVC parameters:", svc_grid_search.best_params_)

# Evaluate the best SVC model on test set
svc_y_pred = svc_grid_search.predict(X_test)
print("SVC Classification Report:\n", classification_report(y_test, svc_y_pred))

# Create a pipeline for Random Forest
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier())
])

# Define hyperparameters to search for Random Forest
rf_param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20]
}

# Perform grid search with cross-validation for Random Forest
rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Save the best Random Forest model
joblib.dump(rf_grid_search.best_estimator_, 'best_rf_model.pkl')

# Print best parameters for Random Forest
print("Best RF parameters:", rf_grid_search.best_params_)

# Evaluate the best Random Forest model on test set
rf_y_pred = rf_grid_search.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred))

# Function to predict sentiment of new posts
def predict_sentiment(text, model):
    return model.predict([text])[0]

# Load the best SVC model
best_svc_model = joblib.load('best_svc_model.pkl')

# Example usage
new_post = "I love this product! It's amazing!"
print(f"Sentiment: {predict_sentiment(new_post, best_svc_model)}")
