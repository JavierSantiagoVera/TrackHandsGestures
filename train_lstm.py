import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from app.model import GestureLSTM

class NpzSeqDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.X = d["X"].astype(np.float32)
        self.y = d["y"].astype(np.int64)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def main():
    ds = NpzSeqDataset("data_lsc_3gestos.npz")  # crea este archivo con tus secuencias
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureLSTM(input_dim=126, hidden_dim=128, num_layers=1, num_classes=3).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(15):
        total, correct, loss_sum = 0, 0, 0.0
        for X, y in dl:
            X = torch.from_numpy(X).to(device) if isinstance(X, np.ndarray) else X.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

        print(f"epoch {epoch+1}: loss={loss_sum/total:.4f} acc={correct/total:.3f}")

    torch.save({"model": model.state_dict()}, "lstm_3gestures.pt")
    print("Saved: lstm_3gestures.pt")

if __name__ == "__main__":
    main()
