# pip install nltk

import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.corpus import treebank
from nltk.wsd import lesk

# Download required datasets (run once)
nltk.download('wordnet')
nltk.download('omw-1.4')   # WordNet multilingual data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')   # ← add this line
nltk.download('words')

# ---------------------------------------------------------------------
# (a) Find Synonyms using WordNet
# ---------------------------------------------------------------------
print("\n(a) Synonyms using WordNet:")
word = "happy"
synonyms = []

for syn in wn.synsets(word):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())

print(f"Synonyms of '{word}':", set(synonyms))

# ---------------------------------------------------------------------
# (b) Find Antonyms using WordNet
# ---------------------------------------------------------------------
print("\n(b) Antonyms using WordNet:")
word = "good"
antonyms = []

for syn in wn.synsets(word):
    for lemma in syn.lemmas():
        # for antonym in lemma.antonyms():
        #     antonyms.append(antonym.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

print(f"Antonyms of '{word}':", set(antonyms))

# ---------------------------------------------------------------------
# (c) Semantic Role Labelling / Named Entity Recognition
# (Here we demonstrate using NLTK's ne_chunk)
# ---------------------------------------------------------------------
print("\n(c) Named Entity Recognition (NER):")
sentence = "Barack Obama was born in Hawaii and served as the President of the United States."
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)
print(named_entities)

# Optional visualization:
# named_entities.draw()

# ---------------------------------------------------------------------
# (d) Word Sense Disambiguation using Lesk Algorithm
# ---------------------------------------------------------------------
print("\n(d) Word Sense Disambiguation:")
sentence2 = "He went to the bank to deposit his money"
ambiguous_word = "bank"
sense = lesk(word_tokenize(sentence2), ambiguous_word)
print(f"Best sense for '{ambiguous_word}':", sense)
print("Definition:", sense.definition())

# ---------------------------------------------------------------------
# (e) Translate Text Using First-Order Logic (FOL)
# We'll symbolically represent logical meaning using nltk.sem.logic
# ---------------------------------------------------------------------
print("\n(e) Translate Text into First-Order Logic:")
from nltk.sem import Expression
read_expr = Expression.fromstring

# Example: "All humans are mortal" → ∀x (Human(x) → Mortal(x))
all_humans_mortal = read_expr('all x. (human(x) -> mortal(x))')

# Example: "Socrates is a human" → Human(Socrates)
socrates_human = read_expr('human(socrates)')

print("FOL 1:", all_humans_mortal)
print("FOL 2:", socrates_human)




# pip install pydatalog


# Import PyDatalog
from pyDatalog import pyDatalog

# Clear any previous logic
pyDatalog.clear()

# Define logic predicates
pyDatalog.create_terms('X, human, mortal')

# Rule: All humans are mortal
mortal(X) <= human(X)

# print(mortal)

# Facts: Socrates is a human, Aristotle is a human
+human('socrates')
+human('aristotle')

# Query: Who is mortal?
print("Who is mortal?")
print(mortal(X))

# Query: Is Socrates mortal?
print("\nIs Socrates mortal?")
print(mortal('socrates'))

# Add another entity that is not human
+human('plato')
+human('confucius')

# See all humans
print("\nAll humans:")
print(human(X))


