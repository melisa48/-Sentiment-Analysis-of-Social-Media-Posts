# Sentiment Analysis of Social Media Posts Project
- This project demonstrates a sentiment analysis pipeline using machine learning techniques. It involves preprocessing text data, training classifiers, and evaluating model performance. The models used include Support Vector Classifier (SVC) and Random Forest Classifier.

## Project Structure
- sentiment_analysis.py: Main script for loading data, training models, evaluating performance, and making predictions.
- sample_data.py: Script to generate synthetic text data with corresponding sentiments.

## Requirements: 
- pandas
- numpy
- scikit-learn
- nltk
- joblib

## Installation
1. Clone the repository:
- git clone <repository_url>
- cd <repository_directory>

2. Install the required packages:
- pip install -r requirements.txt
3. Ensure that you have nltk data installed:
- import nltk
- nltk.download('punkt')
- nltk.download('stopwords')

## Usage
1. Run the Sentiment Analysis Script:
- python sentiment_analysis.py
This script performs the following steps:
- Loads synthetic data generated by `sample_data.py`.
- Splits the data into training and test sets.
- Trains an SVC model and a Random Forest model using grid search for hyperparameter tuning.
- Evaluates the models and prints classification reports.
Saves the best models for future use.
2. Predict Sentiment of New Posts:
Example usage is included in `sentiment_analysis.py`:
- new_post = "I love this product! It's amazing!"
- print(f"Sentiment: {predict_sentiment(new_post, best_svc_model)}")
## Files
`sentiment_analysis.py` 
This script performs the following tasks:
- Imports necessary libraries.
- Loads and splits the dataset.
- Creates pipelines for SVC and Random Forest models.
- Defines hyperparameters for grid search.
- Trains the models and performs grid search with cross-validation.
- Saves the best models.
- Prints the best parameters and classification reports.
- Predicts sentiment for new posts.
`sample_data.py`
This script generates synthetic data for training and testing:
- Creates text samples with positive, negative, and neutral sentiments.
- Returns a pandas DataFrame with the generated data.

## Contributing
- Contributions are welcome! Please fork the repository and submit pull requests with your enhancements.
