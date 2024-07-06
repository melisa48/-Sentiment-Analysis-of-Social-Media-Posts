import pandas as pd
import numpy as np

def create_sample_data(n_samples=1000):
    positive_words = ['love', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful']
    negative_words = ['hate', 'terrible', 'awful', 'horrible', 'disappointing', 'bad']
    neutral_words = ['okay', 'average', 'decent', 'fine', 'mediocre', 'satisfactory']

    texts = []
    sentiments = []

    for _ in range(n_samples):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        if sentiment == 'positive':
            words = np.random.choice(positive_words, size=np.random.randint(3, 7), replace=True)
        elif sentiment == 'negative':
            words = np.random.choice(negative_words, size=np.random.randint(3, 7), replace=True)
        else:
            words = np.random.choice(neutral_words, size=np.random.randint(3, 7), replace=True)
        
        text = ' '.join(words)
        texts.append(text)
        sentiments.append(sentiment)

    return pd.DataFrame({'text': texts, 'sentiment': sentiments})