import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load sample tweets
df = pd.read_csv('data/tweets.csv')
print(df.head())
tweets = df['content'].dropna().tolist()

# NLTK processing
nltk_tokens = [word_tokenize(tweet) for tweet in tweets]
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk_stemmed = [[stemmer.stem(word) for word in tokens] for tokens in nltk_tokens]
nltk_lemmatized = [[lemmatizer.lemmatize(word) for word in tokens] for tokens in nltk_tokens]

# spaCy processing
nlp = spacy.load("en_core_web_sm")
spacy_docs = [nlp(tweet) for tweet in tweets]
spacy_tokens = [[token.text for token in doc] for doc in spacy_docs]
spacy_stemmed = [[token.lemma_ for token in doc] for doc in spacy_docs]
spacy_lemmatized = [[token.lemma_ for token in doc] for doc in spacy_docs]

# Ensure output directory exists
os.makedirs("2-text_processing/data", exist_ok=True)
# Save for visualization
pd.DataFrame({
    "original": tweets,
    "nltk_tokens": nltk_tokens,
    "nltk_lemmas": nltk_lemmatized,
    "nltk_stemmed": nltk_stemmed,
    "spacy_tokens": spacy_tokens,
    "spacy_lemmas": spacy_lemmatized,
    "spacy_stemmed": spacy_stemmed
}).to_csv("data/token_results.csv", index=False)