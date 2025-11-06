# pip install nltk matplotlib wordcloud
import random, re, nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.metrics import precision, recall, f_measure, ConfusionMatrix
from collections import defaultdict, Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Downloads
nltk.download('movie_reviews'); nltk.download('stopwords'); nltk.download('wordnet')

# 1) Load corpus
docs = [
    (list(movie_reviews.words(fid)), cat)
    for cat in movie_reviews.categories()
    for fid in movie_reviews.fileids(cat)
]
random.shuffle(docs)  # 2000 docs total (1000 pos, 1000 neg)

# 2) Preprocess + features
stop = set(stopwords.words('english'))
lemm = WordNetLemmatizer()
token_pat = re.compile(r"[A-Za-z]+")

def normalize(tokens):
    toks = (t.lower() for t in tokens)
    toks = (w for w in toks if token_pat.fullmatch(w) and w not in stop and len(w) > 1)
    toks = [lemm.lemmatize(w) for w in toks]
    return toks

def featurize(tokens):
    toks = normalize(tokens)
    # Presence features; you can extend with bigrams or counts
    return {f"has({w})": True for w in toks}

featuresets = [(featurize(words), label) for (words, label) in docs]

# 3) Split
cut = int(0.8 * len(featuresets))
train_set, test_set = featuresets[:cut], featuresets[cut:]

# 4) Train
clf = nltk.NaiveBayesClassifier.train(train_set)

# 5) Accuracy
acc = nltk.classify.accuracy(clf, test_set)
print(f"Accuracy: {acc:.4f}")

# 6) Detailed metrics using NLTK metrics
# Prepare true and predicted labels
y_true = [label for (_, label) in test_set]
y_pred = [clf.classify(feats) for (feats, _) in test_set]

labels = sorted(set(y_true))  # ['neg','pos']

# Build reference and test sets for NLTK precision/recall/f1
refsets = defaultdict(set)
testsets = defaultdict(set)
for i, (t, p) in enumerate(zip(y_true, y_pred)):
    refsets[t].add(i)
    testsets[p].add(i)

print("\nPer-class metrics:")
for lab in labels:
    p = precision(refsets[lab], testsets[lab])
    r = recall(refsets[lab], testsets[lab])
    f = f_measure(refsets[lab], testsets[lab])
    print(f"  {lab:>3} | Precision: {p if p is not None else 0:.3f} | Recall: {r if r is not None else 0:.3f} | F1: {f if f is not None else 0:.3f}")

# Macro averages
vals = []
for lab in labels:
    p = precision(refsets[lab], testsets[lab]) or 0
    r = recall(refsets[lab], testsets[lab]) or 0
    f = f_measure(refsets[lab], testsets[lab]) or 0
    vals.append((p, r, f))
macro_p = sum(v[0] for v in vals)/len(vals)
macro_r = sum(v[1] for v in vals)/len(vals)
macro_f = sum(v[2] for v in vals)/len(vals)
print(f"\nMacro Avg | Precision: {macro_p:.3f} | Recall: {macro_r:.3f} | F1: {macro_f:.3f}")

# 7) Confusion matrix
cm = ConfusionMatrix(y_true, y_pred)
print("\nConfusion Matrix (counts):")
print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=9))
print("Confusion Matrix (percents):")
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# 8) Most informative features
print("\nMost informative features:")
clf.show_most_informative_features(20)

# 9) Top tokens by class (post-normalization) for inspection
def class_tokens(cat):
    fids = movie_reviews.fileids(cat)
    all_toks = []
    for fid in fids:
        all_toks.extend(normalize(movie_reviews.words(fid)))
    return all_toks

pos_tok = Counter(class_tokens('pos'))
neg_tok = Counter(class_tokens('neg'))
print("\nTop 15 positive tokens:", pos_tok.most_common(15))
print("Top 15 negative tokens:", neg_tok.most_common(15))

# 10) Optional: Word clouds
wc_params = dict(width=1000, height=500, background_color="white", collocations=False)
pos_text = " ".join(" ".join(normalize(movie_reviews.words(fid))) for fid in movie_reviews.fileids('pos'))
neg_text = " ".join(" ".join(normalize(movie_reviews.words(fid))) for fid in movie_reviews.fileids('neg'))
wc_pos = WordCloud(colormap="Greens", **wc_params).generate(pos_text)
wc_neg = WordCloud(colormap="Reds", **wc_params).generate(neg_text)

plt.figure(figsize=(14,6))
plt.subplot(1,2,1); plt.imshow(wc_pos, interpolation="bilinear"); plt.axis("off"); plt.title("Positive reviews")
plt.subplot(1,2,2); plt.imshow(wc_neg, interpolation="bilinear"); plt.axis("off"); plt.title("Negative reviews")
plt.tight_layout(); plt.show()
