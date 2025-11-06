# pip install nltk

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import treebank
from nltk import pos_tag, ne_chunk, FreqDist, word_tokenize
from nltk.chunk import RegexpParser
from nltk.tag import hmm
from heapq import nlargest
import re

# Download required datasets (run once)
nltk.download('punkt')
nltk.download('punkt_tab')     # for newer NLTK versions
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('treebank')
nltk.download('stopwords')

# Sample text
text = """Natural Language Processing enables computers to understand human language.
It combines linguistics, computer science, and artificial intelligence to build intelligent systems."""

# Tokenize the text
tokens = word_tokenize(text)

# ---------------------------------------------------------------------
# (a) Perform Lemmatization
# ---------------------------------------------------------------------
print("\n(a) Lemmatization:")
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token.lower()) for token in tokens]
print(lemmas)

# ---------------------------------------------------------------------
# (b) Perform Stemming
# ---------------------------------------------------------------------
print("\n(b) Stemming:")
stemmer = PorterStemmer()
stems = [stemmer.stem(token.lower()) for token in tokens]
print(stems)

# ---------------------------------------------------------------------
# (c) Identify Parts-of-Speech (POS) using Penn Treebank tagset
# ---------------------------------------------------------------------
print("\n(c) Parts-of-Speech Tagging:")
pos_tags = pos_tag(tokens)
print(pos_tags)

# ---------------------------------------------------------------------
# (d) Implement HMM for POS Tagging
# We'll train a small HMM tagger using the Penn Treebank corpus
# ---------------------------------------------------------------------
print("\n(d) Hidden Markov Model (HMM) POS Tagging:")
train_data = treebank.tagged_sents()[:3000]  # small subset for demo

# Train a simple HMM tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)

# Tag the sample text using the HMM model
hmm_tags = hmm_tagger.tag(tokens)
print(hmm_tags)

# ---------------------------------------------------------------------
# (e) Build a Chunker
# Chunking groups POS-tagged words into meaningful phrases (like Noun Phrases)
# ---------------------------------------------------------------------
print("\n(e) Chunking:")
grammar = r"""
  NP: {<DT>?<JJ>*<NN.*>}     # Noun Phrase
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Verb Phrase
"""
chunk_parser = RegexpParser(grammar)
tree = chunk_parser.parse(pos_tags)
print(tree)

for subtree in tree.subtrees():
    if subtree.label() == "NP":
        print(' '.join(word for word, tag in subtree.leaves()))


# Optional: visualize the chunk tree
# tree.draw()

# ---------------------------------------------------------------------
# (f) Text Summarization (Simple Frequency-based Method)
# ---------------------------------------------------------------------
print("\n(f) Text Summarization:")

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Tokenize sentences and words
sentences = sent_tokenize(text)
words = [word.lower() for word in word_tokenize(text) if word.isalnum()]

# Compute word frequency
fdist = FreqDist(word for word in words if word not in stop_words)

# Score sentences based on word frequencies
sentence_scores = {}
for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in fdist:
            sentence_scores[sent] = sentence_scores.get(sent, 0) + fdist[word]

# Pick top N sentences
summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)

print("Summary:\n", summary)




# pip intall sumy

# Import libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer  # can also try LSA, Luhn, etc.

# Sample text
text = """
Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence.
It enables machines to understand, interpret, and generate human language.
NLP combines computational linguistics, computer science, and deep learning models
to process large amounts of natural language data.
Applications of NLP include machine translation, sentiment analysis, chatbots,
speech recognition, and summarization.
Recent advances in NLP have been driven by transformer-based models such as BERT and GPT.
"""

# Create a parser
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Choose a summarizer
summarizer = LexRankSummarizer()

# Generate summary (specify number of sentences)
summary = summarizer(parser.document, sentences_count=2)

# Print summary
print("📘 Summary:\n")
for sentence in summary:
    print(sentence)
