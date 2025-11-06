# POS tagging using LSTM
# b_pos_lstm.py
import torch, random
import torch.nn as nn

random.seed(0); torch.manual_seed(0)

# Toy POS data: list of (tokens, pos_tags)
data = [
    (["I","love","NLP","!"], ["PRON","VERB","PROPN","PUNCT"]),
    (["You","hate","bugs","!"], ["PRON","VERB","NOUN","PUNCT"]),
    (["NLP","is","great","."], ["PROPN","AUX","ADJ","."]),
]

def build_vocab(seqs):
    from collections import Counter
    cnt = Counter(w for s in seqs for w in s)
    itos = ["<pad>", "<unk>"] + sorted(cnt.keys())
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

w2i, i2w = build_vocab([s for s,_ in data])
t2i, i2t = build_vocab([t for _,t in data])
PAD_W, PAD_T = w2i["<pad>"], t2i["<pad>"]

def encode(batch):
    lens = [len(s) for s,_ in batch]
    L = max(lens)
    X = torch.full((len(batch), L), PAD_W, dtype=torch.long)
    Y = torch.full((len(batch), L), PAD_T, dtype=torch.long)
    for i,(s,t) in enumerate(batch):
        for j,w in enumerate(s):
            X[i,j] = w2i.get(w, w2i["<unk>"])
        for j,lab in enumerate(t):
            Y[i,j] = t2i.get(lab, t2i["<unk>"])
    return X,Y,torch.tensor(lens)

class BiLSTMTagger(nn.Module):
    def __init__(self, vs, ed, hd, nt, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=pad_idx)
        self.lstm = nn.LSTM(ed, hd//2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hd, nt)
    def forward(self, x, lengths):
        e = self.emb(x)
        p = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o,_ = self.lstm(p)
        o,_ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return self.fc(o)

vs, nt = len(i2w), len(i2t)
model = BiLSTMTagger(vs, 100, 128, nt, PAD_W)
lossf = nn.CrossEntropyLoss(ignore_index=PAD_T)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(30):
    X,Y,L = encode(data)
    opt.zero_grad()
    logits = model(X,L)
    loss = lossf(logits.view(-1, nt), Y.view(-1))
    loss.backward(); opt.step()
    if (ep+1)%10==0:
        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = (Y!=PAD_T)
            acc = (pred[mask]==Y[mask]).float().mean().item()
        print(f"Epoch {ep+1} loss={loss.item():.4f} acc={acc:.3f}")

# Test
with torch.no_grad():
    X,Y,L = encode([(["We","love","NLP","!"], ["PRON","VERB","PROPN","PUNCT"])])
    pred = model(X,L).argmax(-1)[0,:L[0]].tolist()
print([i2t[i] for i in pred])
