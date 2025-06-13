'''
word2vec_example.py

Generate Word2Vec embeddings using Gensim on a sample corpus,
identify semantically similar words based on user input, and explain Skip-gram vs CBOW.
'''

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample corpus: list of sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "I love natural language processing",
    "Word2Vec models capture semantic relationships",
    "Gensim makes it easy to train word embeddings",
    "The fox is quick and the dog is lazy"
]

# Preprocess sentences: tokenize and lowercase
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train Word2Vec model
# sg=1 for skip-gram; sg=0 for CBOW
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1  # change to 0 to use CBOW
)

# Get target word from user input
word = input("Enter a word to find similar words: ").strip().lower()

# Identify semantically similar words if in vocabulary
if word in model.wv:
    print(f"Most similar words to '{word}':")
    for similar_word, similarity in model.wv.most_similar(word):
        print(f"{similar_word}: {similarity:.4f}")
else:
    print(f"Word '{word}' not found in vocabulary.")

# Explanation:
# Skip-gram (sg=1) trains the model to predict the context words given a target word.
# It tends to perform better on small datasets and for infrequent words.
# CBOW (sg=0) trains the model to predict the target word given its context words.
# It is faster and tends to do better on larger datasets.