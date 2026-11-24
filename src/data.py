import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MovingMNIST(Dataset):
    def __init__(self, n=800, seq_len=10, size=64, seed=42):
        self.n = n
        self.seq_len = seq_len
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.data = self.make_data()

    def make_data(self):
        all_seqs = []
        for _ in range(self.n):
            frames = []
            #two squares
            p1 = self.rng.randint(8, self.size-18, 2)
            p2 = self.rng.randint(8, self.size-18, 2)
            v1 = self.rng.choice([-2,-1,1,2], 2)
            v2 = self.rng.choice([-2,-1,1,2], 2)

            for t in range(self.seq_len):
                img = np.zeros((self.size, self.size), np.float32)
                self.put_square(img, p1, 0.75)
                self.put_square(img, p2, 0.45)
                frames.append(img)

                p1 = p1 + v1
                p2 = p2 + v2
                # bounce
                for i in range(2):
                    if p1[i] < 4 or p1[i] > self.size-16:
                        v1[i] *= -1
                    if p2[i] < 4 or p2[i] > self.size-16:
                        v2[i] *= -1
            all_seqs.append(np.stack(frames))
        return np.array(all_seqs)

    def put_square(self, img, pos, val):
        s= 11
        x, y = int(pos[0]), int(pos[1])
        img[y:y+s, x:x+s] = val

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        seq = self.data[i]
        #5 in, 5 out
        x = seq[:5]
        y = seq[5:]
        return torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y).unsqueeze(1)

def get_loaders(bs=16, n_train=600, n_val=120):
    train = MovingMNIST(n=n_train, seed=42)
    val = MovingMNIST(n=n_val, seed=99)
    return DataLoader(train, bs, shuffle=True), DataLoader(val, bs)
