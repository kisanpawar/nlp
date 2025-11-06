# pip install nltk

# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')   # ← Add this line for nltk>= 3.9
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = """Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence. 
It enables machines to understand, interpret, and generate human language."""

# ---------------------------------------------------------------------
# (a) Convert the Text into Tokens
# ---------------------------------------------------------------------
print("\n(a) Tokenization:")
word_tokens = word_tokenize(text)
print(word_tokens)

# ---------------------------------------------------------------------
# (b) Find the Word Frequency
# ---------------------------------------------------------------------
print("\n(b) Word Frequency:")
fdist = FreqDist(word_tokens)
for word, freq in fdist.most_common():
    print(f"{word} : {freq}")

# ---------------------------------------------------------------------
# (c) Demonstrate a Bigram Language Model
# ---------------------------------------------------------------------
print("\n(c) Bigram Model:")
bi_grams = list(bigrams(word_tokens))
print(bi_grams)

# ---------------------------------------------------------------------
# (d) Demonstrate a Trigram Language Model
# ---------------------------------------------------------------------
print("\n(d) Trigram Model:")
tri_grams = list(trigrams(word_tokens))
print(tri_grams)

# ---------------------------------------------------------------------
# (e) Generate Regular Expression for a Given Text
# Example: extract words starting with capital letters
# ---------------------------------------------------------------------
print("\n(e) Regular Expression Example:")
pattern = r'\b[A-Z][a-zA-Z]+\b'
capitalized_words = re.findall(pattern, text)
print("Words starting with capital letters:", capitalized_words)

# ---------------------------------------------------------------------
# (f) Text Normalization
# Includes: lowercasing, removing punctuation, stopwords, and lemmatization
# ---------------------------------------------------------------------
print("\n(f) Text Normalization:")
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text.lower())

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("Normalized Tokens:", lemmatized_tokens)
