#Practical 1(a): Convert the Text into Tokens

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


import re


textMessage = "Welcome to NLP Programming"

tokens = word_tokenize(textMessage)


print(tokens)  # Output: ['Welcome', 'to', 'NLP', 'Programming']
# Output: ['Welcome', 'to', 'NLP', 'Programming']

# 2 Find the Word Frequency

mesgText = "Welcome to NLP Programming. NLP is an interesting field. Welcome again to the world of NLP."

frequencyTest = word_tokenize(mesgText)

freq = nltk.FreqDist(frequencyTest)

print("Frquency Common Word",freq.most_common(3))  



# Output: [('NLP', 3), ('to', 2), ('Welcome', 2)]
# Output: [('NLP', 3), ('to', 2), ('Welcome', 2)]

# 3 Demonstrate a bigram Language Model.

biagRamText = "Welcome to NLP Programming. NLP is an interesting field."

token = word_tokenize(biagRamText)

bigrams = list(nltk.bigrams(token))

print("Bigrams:",bigrams)
# Output: Bigrams: [('Welcome', 'to'), ('to', 'NLP'),



# 3 Demonstrate a Trigram Language Model;

triGram = "Welcome to NLP Programming. NLP is an interesting field."

token = word_tokenize(triGram)

triGramsValue = list(nltk.trigrams(token))

print("Trigram:>>>>>>>>>>>",triGramsValue)
# Output: Bigrams: [('Welcome', 'to'), ('to', 'NLP'),



# 4 Generate Regular Expression for a Given Text.

textEmail  = "Please contact us at  kisan@gmail.com ramesh@gmail.com for further information."
emailPattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails = re.findall(emailPattern, textEmail)
print("Email Address Found:",emails)

# Output: Email



# F. Text Normalization
import re
import unicodedata

def abc(text):
    # Convert text to lowercase
    normalized_text = text.lower()
    
    # Remove special characters and punctuation
    normalized_text = re.sub(r'[^\w\s]', '', normalized_text)
    
    # Normalize unicode characters (e.g., accented letters)
    normalized_text = unicodedata.normalize('NFKD', normalized_text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove extra whitespaces
    normalized_text = " ".join(normalized_text.split())
    
    return normalized_text

# Input from user
input_text = input("Enter Text To Normalize: ")

# Normalize the text
normalized_result = abc(input_text)

# Output result
print("Normalized text:", normalized_result)














