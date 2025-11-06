# A) Find the synonym of a word using WordNet

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

word = "Good"
synonyms = []

for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())

synonyms = list(set(synonyms))
print("Synonyms for", word + ":")
print(synonyms)


#

# B) Find the antonym of a word

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

word = "Good"
antonyms = []

for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        for antonym in lemma.antonyms():
            antonyms.append(antonym.name())

antonyms = list(set(antonyms))
print("Antonyms for", word + ":")
print(antonyms)







# C) Implement Semantic Role Labelling to Identify Named Entities

import nltk
import spacy

# Download necessary NLTK data
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Apple Inc. was founded by Steve Jobs and Steve Wozinak in Cupertino, California"

# Process the text
doc = nlp(text)

# Extract entities and their labels
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Print entities with their labels
for entity, label in entities:
    print(f"Entity: {entity}, Label: {label}")
    
    

# D) Resolve the ambiguity
import nltk
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example sentence
text = "The Chicken is Ready to Eat"

# Process the text
doc = nlp(text)

# Display tokens, POS tags, and lemmas (base forms)
for token in doc:
    print(f"Token: {token.text}, POS: {token.pos_}, Sense: {token.lemma_}")



# E) Translate the text using First-order logic
from pyDatalog import pyDatalog

# Declare the necessary facts
pyDatalog.create_terms('human, mortal, X')

# Define some facts
+human('John')
+human('Alice')

# Define the logical rule: All humans are mortal
mortal(X) <= human(X)

# Query the rule
result = mortal(X)

# Check the result and print output
if result:
    print("All humans are mortal")
else:
    print("Not all humans are mortal")
