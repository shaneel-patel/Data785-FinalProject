import torch
import torch.nn as nn

# small unet for diffusion
class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(6, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.c3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.u3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.u2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        e1 = self.act(self.c1(x))
        e2 = self.act(self.c2(e1))
        e3 = self.act(self.c3(e2))
        d = self.act(self.u3(e3))
        d = self.act(self.u2(d + e2))
        return self.out(d + e1)

class Diffusion(nn.Module):
    def __init__(self, T=40):
        super().__init__()
        self.net = SmallUNet()
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0))

    def forward(self, noisy, t, cond):
        cond = cond.squeeze(2)
        if noisy.dim() == 5:
            noisy = noisy.squeeze(1)
        x = torch.cat([cond, noisy.unsqueeze(1)], 1)
        return self.net(x)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].sqrt().view(-1,1,1,1)
        b = (1 - self.alphas_cumprod[t]).sqrt().view(-1,1,1,1)
        return a * x0 + b * noise, noise

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64*64, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 64*64)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(2).view(B, 5, -1)
        h = torch.relu(self.fc1(x))
        out, _ = self.lstm(h)
        pred = self.fc2(out[:, -1])
        return pred.view(B, 1, 64, 64).unsqueeze(1)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(64*64, 128)
        enc = nn.TransformerEncoderLayer(128, 4, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, 2)
        self.head = nn.Linear(128, 64*64)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(2).view(B, 5, -1)
        h = self.embed(x)
        h = self.tf(h)
        pred = self.head(h[:, -1])
        return pred.view(B, 1, 64, 64).unsqueeze(1)
