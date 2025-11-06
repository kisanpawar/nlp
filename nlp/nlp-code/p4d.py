# d_wsd_lstm.py
import torch, random
import torch.nn as nn

random.seed(7); torch.manual_seed(7)

# Toy WSD: for lemma "bank" with two senses: 0=financial_institution, 1=river_side
# Each example: (tokens, target_index, gold_sense)
train_data = [
    (["I","went","to","the","bank","to","deposit","money"], 4, 0),
    (["The","boat","reached","the","bank","of","the","river"], 4, 1),
    (["She","works","at","a","bank","downtown"], 5-1, 0),
    (["Children","played","on","the","bank","near","the","bridge"], 4, 1),
]

def build_vocab(seqs):
    from collections import Counter
    cnt = Counter(w for s,_,_ in train_data for w in s)
    itos = ["<pad>","<unk>"] + sorted(cnt.keys())
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

w2i, i2w = build_vocab([s for s,_,_ in train_data])
PAD = w2i["<pad>"]

def encode_batch(batch):
    lens = [len(s) for s,_,_ in batch]
    L = max(lens)
    B = len(batch)
    X = torch.full((B,L), PAD, dtype=torch.long)
    idx = torch.zeros(B, dtype=torch.long)
    y  = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B,L, dtype=torch.bool)
    for i,(s,ti,lab) in enumerate(batch):
        for j,w in enumerate(s):
            X[i,j] = w2i.get(w, w2i["<unk>"])
            mask[i,j] = True
        idx[i] = ti
        y[i] = lab
    return X, idx, y, lens, mask

class WSDModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_senses, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.enc = nn.LSTM(emb_dim, hid_dim//2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim, num_senses)
    def forward(self, x, lengths, target_idx):
        e = self.emb(x)
        p = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        h,_ = self.enc(p)
        h,_ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        # Gather hidden at target positions
        bsz = h.size(0)
        gather = h[torch.arange(bsz), target_idx]
        logits = self.fc(gather)
        return logits

num_senses = 2
vs = len(i2w)
model = WSDModel(vs, 100, 128, num_senses, PAD)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

for ep in range(200):
    X, idx, y, lengths, mask = encode_batch(train_data)
    lengths = torch.tensor(lengths, dtype=torch.long)
    opt.zero_grad()
    logits = model(X, lengths, idx)
    loss = lossf(logits, y)
    loss.backward(); opt.step()
    if (ep+1)%50==0:
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = (pred==y).float().mean().item()
        print(f"Epoch {ep+1} loss={loss.item():.4f} acc={acc:.3f}")

# Test
test = [(["We","sat","on","the","bank","watching","the","ducks"], 4, 1)]
X, idx, y, lengths, mask = encode_batch(test)
lengths = torch.tensor(lengths, dtype=torch.long)
with torch.no_grad():
    logits = model(X, lengths, idx)
    pred = logits.argmax(-1).item()
print("Predicted sense:", pred, "(1=river_side, 0=financial_institution)")
