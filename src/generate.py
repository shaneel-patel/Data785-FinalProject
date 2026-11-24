import torch
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(__file__))
from data import get_loaders
from models import RNN, Transformer

def plot_stuff(cond, pred, real, name="result"):
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        ax[0,i].imshow(cond[0,i,0].cpu(), cmap="gray")
        ax[0,i].set_title(f"in {i}")
        ax[0,i].axis("off")
        ax[1,i].imshow(real[0,i,0].cpu(), cmap="gray")
        ax[1,i].set_title(f"real {i}")
        ax[1,i].axis("off")
    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(3,3))
    plt.imshow(pred[0,0,0].cpu().detach(), cmap="gray")
    plt.title("predicted next frame")
    plt.axis("off")
    plt.show()

def run(model_name="rnn"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loader = get_loaders(bs=1)
    cond, real = next(iter(loader))
    cond, real = cond.to(device), real.to(device)

    if model_name == "rnn":
        m = RNN().to(device)
        m.load_state_dict(torch.load("../artifacts/rnn.pt", map_location=device))
    else:
        m = Transformer().to(device)
        m.load_state_dict(torch.load("../artifacts/tf.pt", map_location=device))

    m.eval()
    with torch.no_grad():
        pred = m(cond)

    plot_stuff(cond, pred, real, model_name)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="rnn")
    args = p.parse_args()
    run(args.model)
