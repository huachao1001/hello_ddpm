
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from networks import ContextUnet
from utils import save_png, save_gif
from ddpm import DDPM


class Train():
    def __init__(self, batch_size=128, nclasses=10, epochs=20, steps=400, lr=1e-4, save_dir='./logs', drop_prob=0.1):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.steps = steps
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.epochs = epochs
        self.nclasses = nclasses
        self.save_dir = save_dir
        self.dataloader = self.init_data()
        self.loss_ema = None
        self.ddpm = DDPM(1e-4, 0.02, steps, self.device)
        self.model = ContextUnet(
            in_channels=1, n_feat=128, n_classes=nclasses).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_mse = torch.nn.MSELoss()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def init_data(self):
        # mnist is already normalised 0 to 1
        tf = transforms.Compose([transforms.ToTensor()])
        dataset = MNIST("./data", train=True, download=True, transform=tf)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=5)
        return dataloader

    def eval(self, epoch):
        self.model.eval()
        print("eval...")
        cnt_per_digit = 4  # 每个数字生成cnt_per_digit张图
        # c = torch.arange(0, 10).to(self.device)
        # c = c.repeat(cnt_per_digit)
        # batch = cnt_per_digit*10
        numbers = [i % 10 for i in range(cnt_per_digit*10)]

        with torch.no_grad():
            for w in [0.0, 0.5, 2.0]:
                x, x_gen_store = self.sample(numbers, w)
                if len(x_gen_store) == 0:
                    continue
                x_all = torch.permute(x, (0, 2, 3, 1)).cpu().numpy()
                x_all = np.clip(x_all*255, 0, 255).astype(np.uint8)
                save_png(x_all, cnt_per_digit*2, 10, os.path.join(
                    self.save_dir, f"image_ep{epoch}_w{w}.png"))

                if epoch % 5 == 0 or epoch == self.epochs-1:
                    dst = os.path.join(
                        self.save_dir,  f"gif_ep{epoch}_w{w}.gif")
                    save_gif(x_gen_store, cnt_per_digit, 10, dst)

    def infer(self, x, c, step):
        # print(x.shape, c.shape, step.shape)
        x, c = x.to(self.device), c.to(self.device)
        mask = torch.zeros_like(c)
        if self.model.training:
            x_noisy, noise = self.ddpm.add_noise(x, step)
            mask = torch.bernoulli(mask+self.drop_prob)
            y = self.model(x_noisy,  c, step / self.steps, mask)
            return y, noise
        else:
            # print("eval.....")
            y = self.model(x,  c, step / self.steps, mask)
        return y

    def unorm(self, x):
        x = x.detach().cpu().numpy()
        v_min, v_max = np.min(-x), np.max(-x)
        x = np.clip(255*(x-v_min)/(v_max-v_min), 0, 255).astype(np.uint8)
        x = np.transpose(x, (0, 2, 3, 1))
        return x

    def sample(self, number_arr: list, guide_w):
        batch = len(number_arr)
        c = torch.tensor(number_arr).to(self.device)
        x = torch.randn(batch, 1, 28, 28).to(self.device)
        x_i_store = []
        for i in range(self.steps-1, -1, -1):
            step = torch.tensor([i]).to(self.device)
            step = step.repeat(batch)
            if guide_w == 0:
                noise = self.infer(x, c, step)
            else:
                # double batch
                x = x.repeat(2, 1, 1, 1)
                step = step.repeat(2)
                c = c.repeat(2)

                noise = self.infer(x, c, step)
                eps1, eps2 = noise[:batch], noise[batch:]
                noise = (1+guide_w)*eps1 - guide_w*eps2
                x = x[:batch]
                c = c[:batch]

            x = self.ddpm.denoise(x, i, noise)
            if i % 20 == 0 or i == self.steps or i < 8:
                img = self.unorm(x)
                x_i_store.append(img)
        if len(x_i_store) > 0:
            x_i_store = np.array(x_i_store)
        return x, x_i_store

    def train_step(self, epoch, i, x, c):
        self.optim.zero_grad()
        step = torch.randint(1, self.steps, (x.shape[0],)).to(self.device)
        y, noise = self.infer(x, c, step)
        loss = self.loss_mse(noise, y)
        loss.backward()
        if self.loss_ema is None:
            self.loss_ema = loss.item()
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * loss.item()
        self.optim.step()
        return self.loss_ema

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.dataloader)
        for i, (x, c) in enumerate(pbar):
            loss_ema = self.train_step(epoch, i, x, c)
            pbar.set_description(f"epoch:{epoch:d}, loss: {loss_ema:.4f}")
        self.eval(epoch)

    def train(self):
        # self.eval(0)
        for epoch in range(self.epochs):
            self.train_epoch(epoch)


Train().train()
