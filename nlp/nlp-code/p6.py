# pip install nltk pandas
import re, random, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')

# 1) Load labeled Flipkart reviews CSV (columns: review_text, sentiment in {positive, negative, neutral})
df = pd.read_csv("flipkart_reviews.csv")  # use a Kaggle dataset with labels
df = df.dropna(subset=["review_text", "sentiment"])

# 2) NLTK preprocessing
stop = set(stopwords.words('english'))
lemm = WordNetLemmatizer()
token_pat = re.compile(r"[A-Za-z]+")

def tokenize(text):
    toks = token_pat.findall(text.lower())
    toks = [t for t in toks if t not in stop and len(t) > 1]
    toks = [lemm.lemmatize(t) for t in toks]
    return toks

def featurize(text):
    toks = tokenize(text)
    return {f"has({w})": True for w in toks}

# 3) Build features and split
data = [(featurize(t), y) for t, y in zip(df["review_text"], df["sentiment"])]
random.shuffle(data)
cut = int(0.8*len(data))
train_set, test_set = data[:cut], data[cut:]

# 4) Train and evaluate
clf = nltk.NaiveBayesClassifier.train(train_set)
acc = nltk.classify.accuracy(clf, test_set)
print("Accuracy:", acc)
clf.show_most_informative_features(20)

# 5) Predict
def predict(texts):
    return [clf.classify(featurize(t)) for t in texts]

print(predict(["Great battery life and camera!", "Terrible packaging, item damaged."]))







# pip install nltk pandas matplotlib wordcloud

import re, nltk, pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

nltk.download('stopwords')

# 1) Load your CSV created earlier
df = pd.read_csv("flipkart_reviews.csv", encoding="utf-8")

# 2) Build stopword set and simple cleaner
stop = set(stopwords.words('english'))
extra = {"flipkart", "phone", "product", "good", "bad"}  # optional domain words to drop
stop |= extra

def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join([w for w in s.split() if w not in stop and len(w) > 2])

# 3) Create corpora by sentiment
pos_corpus = " ".join(df.loc[df["sentiment"] == "positive", "review_text"].astype(str).map(clean_text))
neg_corpus = " ".join(df.loc[df["sentiment"] == "negative", "review_text"].astype(str).map(clean_text))
all_corpus = " ".join(df["review_text"].astype(str).map(clean_text))

# 4) Generate word clouds
wc_params = dict(width=900, height=500, background_color="white", collocations=False)

wc_all = WordCloud(**wc_params).generate(all_corpus)
wc_pos = WordCloud(colormap="Greens", **wc_params).generate(pos_corpus)
wc_neg = WordCloud(colormap="Reds", **wc_params).generate(neg_corpus)

# 5) Plot
plt.figure(figsize=(14, 10))

plt.subplot(3,1,1)
plt.imshow(wc_all, interpolation="bilinear")
plt.axis("off")
plt.title("All Reviews Word Cloud")

plt.subplot(3,1,2)
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews Word Cloud")

plt.subplot(3,1,3)
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews Word Cloud")

plt.tight_layout()
plt.show()
