# pip install nltk matplotlib wordcloud
import random, re, nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('movie_reviews'); nltk.download('stopwords'); nltk.download('wordnet')

# 1) Prepare documents: list of (tokens, label)
docs = [
    (list(movie_reviews.words(fid)), cat)
    for cat in movie_reviews.categories()
    for fid in movie_reviews.fileids(cat)
]
random.shuffle(docs)  # preserves label balance

# 2) Preprocess and featurize for NLTK
stop = set(stopwords.words('english'))
lemm = WordNetLemmatizer()
token_pat = re.compile(r"[A-Za-z]+")

def normalize(tokens):
    toks = [t.lower() for t in tokens]
    toks = [w for w in toks if token_pat.fullmatch(w) and w not in stop and len(w) > 1]
    toks = [lemm.lemmatize(w) for w in toks]
    return toks

def featurize(tokens):
    toks = normalize(tokens)
    return {f"has({w})": True for w in toks}

featuresets = [(featurize(words), label) for (words, label) in docs]

# 3) Train/test split
cut = int(0.8 * len(featuresets))
train_set, test_set = featuresets[:cut], featuresets[cut:]

# 4) Train NLTK Naive Bayes
clf = nltk.NaiveBayesClassifier.train(train_set)

# 5) Evaluate
acc = nltk.classify.accuracy(clf, test_set)
print("Accuracy:", acc)
clf.show_most_informative_features(20)







# pip install wordcloud matplotlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Build corpora by label using original tokens -> normalized strings
pos_text = " ".join(" ".join(normalize(movie_reviews.words(fid)))
                    for fid in movie_reviews.fileids('pos'))
neg_text = " ".join(" ".join(normalize(movie_reviews.words(fid)))
                    for fid in movie_reviews.fileids('neg'))

wc_params = dict(width=1000, height=500, background_color="white", collocations=False)
wc_pos = WordCloud(colormap="Greens", **wc_params).generate(pos_text)
wc_neg = WordCloud(colormap="Reds", **wc_params).generate(neg_text)

plt.figure(figsize=(14,6))
plt.subplot(1,2,1); plt.imshow(wc_pos, interpolation="bilinear"); plt.axis("off"); plt.title("Positive reviews")
plt.subplot(1,2,2); plt.imshow(wc_neg, interpolation="bilinear"); plt.axis("off"); plt.title("Negative reviews")
plt.tight_layout(); plt.show()
