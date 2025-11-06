# c_ner_bilstm_crf.py
# pip install torch==2.2.0
import torch, random
import torch.nn as nn

random.seed(1); torch.manual_seed(1)

# Tiny toy BIO data
train_data = [
    (["John","lives","in","New","York","."], ["B-PER","O","O","B-LOC","I-LOC","O"]),
    (["Mary","works","at","Google","."],     ["B-PER","O","O","B-ORG","O"]),
]

def build_vocab(seqs):
    from collections import Counter
    cnt = Counter(w for s in seqs for w in s)
    itos = ["<pad>", "<unk>"] + sorted(cnt.keys())
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

w2i, i2w = build_vocab([s for s,_ in train_data])
t2i, i2t = build_vocab([t for _,t in train_data])
PAD_W, PAD_T = w2i["<pad>"], t2i["<pad>"]

def encode(batch):
    lens = [len(s) for s,_ in batch]
    L = max(lens)
    X = torch.full((len(batch), L), PAD_W, dtype=torch.long)
    Y = torch.full((len(batch), L), PAD_T, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.bool)
    for i,(s,t) in enumerate(batch):
        for j,w in enumerate(s):
            X[i,j] = w2i.get(w, w2i["<unk>"])
            Y[i,j] = t2i.get(t[j], t2i["<unk>"])
            mask[i,j] = True
    return X,Y,mask,lens

class BiLSTM_CRF(nn.Module):
    def __init__(self, vs, ed, hd, nt, pad_idx, pad_tag):
        super().__init__()
        self.nt = nt
        self.pad_tag = pad_tag
        self.emb = nn.Embedding(vs, ed, padding_idx=pad_idx)
        self.lstm = nn.LSTM(ed, hd//2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hd, nt)
        self.transitions = nn.Parameter(torch.randn(nt, nt))  # [to, from]
        self.transitions.data[pad_tag, :] = -1e4
        self.transitions.data[:, pad_tag] = -1e4

    def _log_sum_exp(self, x, dim=-1):
        m, _ = torch.max(x, dim=dim, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))

    def forward(self, x, mask):
        e = self.emb(x)
        o,_ = self.lstm(e)
        logits = self.fc(o)  # [B,T,nt]
        return logits

    def neg_log_likelihood(self, logits, tags, mask):
        B,T,nt = logits.shape
        start = torch.full((B,1,nt), -1e4, device=logits.device)
        start[:,:, :] = -1e4
        alpha = logits[:,0]  # first timestep
        for t in range(1, T):
            emit = logits[:,t].unsqueeze(2)             # [B,nt,1]
            trans = self.transitions.unsqueeze(0)       # [1,nt,nt]
            score = alpha.unsqueeze(1) + trans + emit   # [B,nt,nt]
            alpha = self._log_sum_exp(score, dim=2).squeeze(2)
            m = mask[:,t].unsqueeze(1)
            alpha = torch.where(m, alpha, alpha)  # masking keeps states
        logZ = self._log_sum_exp(alpha, dim=1).squeeze(1)

        # Gold path score
        score = torch.zeros(B, device=logits.device)
        for b in range(B):
            prev = None
            for t in range(T):
                if not mask[b,t]: break
                tag = tags[b,t]
                score[b] += logits[b,t,tag]
                if prev is not None:
                    score[b] += self.transitions[tag, prev]
                prev = tag
        return (logZ - score).mean()

    def viterbi_decode(self, logits, mask):
        B,T,nt = logits.shape
        backpointers = []
        viterbi = logits[:,0]
        for t in range(1,T):
            trans = self.transitions.unsqueeze(0)       # [1,nt,nt]
            score = viterbi.unsqueeze(1) + trans + logits[:,t].unsqueeze(2)
            best_score, best_path = torch.max(score, dim=2)
            viterbi = best_score
            backpointers.append(best_path)
        best_tags = []
        for b in range(B):
            L = mask[b].sum().item()
            last = torch.argmax(viterbi[b]).item()
            path = [last]
            for bp in reversed(backpointers[:L-1]):
                last = bp[b, last].item()
                path.append(last)
            path.reverse()
            best_tags.append(path)
        return best_tags

vs, nt = len(i2w), len(i2t)
model = BiLSTM_CRF(vs, 100, 128, nt, PAD_W, PAD_T)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)

for ep in range(200):
    X,Y,mask,lens = encode(train_data)
    logits = model(X, mask)
    loss = model.neg_log_likelihood(logits, Y, mask)
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1)%50==0:
        print(f"Epoch {ep+1} loss={loss.item():.4f}")

# Decode
model.eval()
X,Y,mask,lens = encode([(["Mary","lives","in","Paris","."], ["B-PER","O","O","B-LOC","O"])])
with torch.no_grad():
    logits = model(X, mask)
    pred_paths = model.viterbi_decode(logits, mask)
pred_tags = [i2t[i] for i in pred_paths[0]]
print("Prediction:", pred_tags)
