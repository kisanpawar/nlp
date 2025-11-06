# Demonstrate a Trigram Language Model
# Perform Lemmatization
import nltk
nltk.download('punkt')
nltk.download('omw-1.4') 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')

#intlize the lemmatizer
lemmatizer = WordNetLemmatizer()
text = "The striped bats are hanging on their feet for best"
tokens = word_tokenize(text)
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized Words:", lemmatized_words)


#2 Perform Stemming:
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
text2 = "running runner runs easily fairies"
tokens2 = word_tokenize(text2)
stemmed_words = [stemmer.stem(token) for token in tokens2]
print("Stemmed Words:", stemmed_words)



#3 Identify Parts-of-Speech using Penn Treebank tag Set

# part of speech  
# dt - determiner
# jj - adjective
# nn - noun, singular or mass 
# nns - noun plural
# vbz - verb, 3rd person singular present
# in - preposition or subordinating conjunction



import nltk
nltk.download('averaged_perceptron_tagger')
text3 = "The quick brown fox jumps over the lazy dog"
tokens3 = word_tokenize(text3)
pos_tags = nltk.pos_tag(tokens3)
print("Parts-of-Speech Tags:", pos_tags)


# C - Implement HMM for Pos Tagging
# D. Implement HMM for POS Tagging
import nltk
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('treebank')

# Load the Penn Treebank corpus
corpus = nltk.corpus.treebank

# Split the corpus into tagged sentences
tagged_sentences = corpus.tagged_sents()

# Split data into training and testing sets
random.seed(123)  # for reproducibility
split_ratio = 0.8
split_index = int(len(tagged_sentences) * split_ratio)
training_sentences = tagged_sentences[:split_index]
testing_sentences = tagged_sentences[split_index:]

# Create and train HMM POS tagger
from nltk.tag import hmm

# Train the HMM model on training data
hmm_tagger = hmm.HiddenMarkovModelTrainer().train(training_sentences)

# Evaluate the model on testing data
accuracy = hmm_tagger.evaluate(testing_sentences)
print("HMM POS Tagger Accuracy:", accuracy)

# Tag a new sentence
new_sentence = "This is a test sentence"
new_words = nltk.word_tokenize(new_sentence)
predicted_tags = hmm_tagger.tag(new_words)

print("Predicted POS tags for the new sentence:")
print(predicted_tags)


# D Build a Chunker:


# E) Build a Chunk
import nltk

# Sample sentence
L = "The quick brown fox jumps over the lazy dog"

# Tokenize the sentence
words = nltk.word_tokenize(L)

# Perform POS tagging
pos_tags = nltk.pos_tag(words)

# Define a grammar for NP (Noun Phrase)
grammar = r"""NP: {<DT>?<JJ>*<NN.*>+}"""

# Create a chunk parser
chunk_parser = nltk.RegexpParser(grammar)

# Parse the POS tagged sentence
chunks_sentence = chunk_parser.parse(pos_tags)

# Print noun phrases
for subtree in chunks_sentence.subtrees():
    if subtree.label() == 'NP':
        print(' '.join(word for word, tag in subtree.leaves()))
# Print the lemmatized words
print("Lemmatized Words:", lemmatized_words)

# Text Summarization:


# Text Summarization

# Install sumy library (uncomment this line if not installed)
# !pip install sumy

import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Input text for summarization
text = """Text summarization is the process of generating short, fluent, and most importantly 
accurate summary of a respectively longer text document. The main idea behind automatic text 
summarization is to be able to find a short subset of the most essential information from the 
entire set and present it in a human-readable format. As online textual data grows, automatic 
text summarization methods have the potential to be very helpful because more useful information 
can be read in a short time."""

# Parse and tokenize the text
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Create LSA summarizer
summarizer = LsaSummarizer()

# Ask user for number of sentences
sentences_count = int(input("Enter the number of summary sentences: "))

# Generate summary
summary = summarizer(parser.document, sentences_count)

# Print summarized sentences
for sentence in summary:
    print(sentence)
# Print the stemmed words