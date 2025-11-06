# a_sequence_labeling.py
# pip install torch==2.2.0
import torch, random
import torch.nn as nn
from typing import List, Tuple

random.seed(42)
torch.manual_seed(42)

# 1) Toy data: tokens and labels of equal length per sentence
toy_sentences = [
    ["John", "loves", "NLP"],
    ["Mary", "hates", "bugs"],
    ["NLP", "is", "fun"],
]
toy_labels = [
    ["PER", "O", "O"],
    ["PER", "O", "O"],
    ["O", "O", "O"],
]

# Build vocabularies
def build_vocab(seqs: List[List[str]], min_freq=1):
    from collections import Counter
    cnt = Counter(w for s in seqs for w in s)
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.items() if c >= min_freq]
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

word2id, id2word = build_vocab(toy_sentences)
tag2id, id2tag = build_vocab(toy_labels)
PAD_WORD = word2id["<pad>"]
PAD_TAG = tag2id["<pad>"]

def encode_batch(batch: List[Tuple[List[str], List[str]]]):
    lens = [len(s) for s,_ in batch]
    maxlen = max(lens)
    x = torch.full((len(batch), maxlen), PAD_WORD, dtype=torch.long)
    y = torch.full((len(batch), maxlen), PAD_TAG, dtype=torch.long)
    for i,(s,t) in enumerate(batch):
        for j, w in enumerate(s):
            x[i,j] = word2id.get(w, word2id["<unk>"])
        for j, lab in enumerate(t):
            y[i,j] = tag2id.get(lab, tag2id["<unk>"])
    lengths = torch.tensor(lens, dtype=torch.long)
    return x, y, lengths

dataset = list(zip(toy_sentences, toy_labels))
random.shuffle(dataset)
train = dataset

# 2) Model: BiLSTM token classifier
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_tags, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim//2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim, num_tags)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)
        return logits

vocab_size = len(id2word)
num_tags = len(id2tag)
model = BiLSTMTagger(vocab_size, emb_dim=100, hid_dim=128, num_tags=num_tags, pad_idx=PAD_WORD)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3) Train
for epoch in range(20):
    model.train()
    x, y, lengths = encode_batch(train)
    optim.zero_grad()
    logits = model(x, lengths)               # [B, T, num_tags]
    loss = criterion(logits.view(-1, num_tags), y.view(-1))
    loss.backward()
    optim.step()
    if (epoch+1) % 5 == 0:
        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = (y != PAD_TAG)
            acc = (pred[mask] == y[mask]).float().mean().item()
        print(f"Epoch {epoch+1} | loss={loss.item():.4f} | token-acc={acc:.3f}")

# 4) Inference
model.eval()
x, y, lengths = encode_batch([(["John","hates","NLP"], ["PER","O","O"])])
with torch.no_grad():
    logits = model(x, lengths)
    pred = logits.argmax(-1)[0, :lengths[0]].tolist()
print("Prediction:", [id2tag[i] for i in pred])
