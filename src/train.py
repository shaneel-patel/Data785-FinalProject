import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(__file__))
from data import get_loaders
from models import RNN, Transformer, Diffusion

def train_rnn(epochs=4, device="cpu"):
    model = RNN().to(device)
    opt = Adam(model.parameters(), lr=0.001)
    loader, _ = get_loaders()
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        for cond, tgt in tqdm(loader, desc=f"rnn {ep}"):
            cond, tgt = cond.to(device), tgt.to(device)
            pred = model(cond)
            loss = F.mse_loss(pred, tgt[:,0:1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(ep, loss_sum/len(loader))
    torch.save(model.state_dict(), "../artifacts/rnn.pt")

def train_tf(epochs=4, device="cpu"):
    model = Transformer().to(device)
    opt = Adam(model.parameters(), lr=0.001)
    loader, _ = get_loaders()
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        for cond, tgt in tqdm(loader, desc=f"tf {ep}"):
            cond, tgt = cond.to(device), tgt.to(device)
            pred = model(cond)
            loss = F.mse_loss(pred, tgt[:,0:1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(ep, loss_sum/len(loader))
    torch.save(model.state_dict(), "../artifacts/tf.pt")

def train_diff(epochs=3, device="cpu"):
    model = Diffusion().to(device)
    opt = Adam(model.parameters(), lr=0.001)
    loader, _ = get_loaders(bs=8)
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        for cond, tgt in tqdm(loader, desc=f"diff {ep}"):
            cond, tgt = cond.to(device), tgt.to(device)
            x0 = tgt[:,0]
            t = torch.randint(0, model.T, (x0.size(0),), device=device)
            noisy, noise = model.add_noise(x0, t)
            pred = model(noisy, t, cond)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(ep, loss_sum/len(loader))
    torch.save(model.state_dict(), "../artifacts/diff.pt")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="rnn")
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("using", dev)
    if args.model == "rnn":
        train_rnn(args.epochs, dev)
    elif args.model == "transformer":
        train_tf(args.epochs, dev)
    else:
        train_diff(args.epochs, dev)
