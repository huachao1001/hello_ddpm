
import torch
import torch.nn as nn
import numpy as np


class DDPM():
    def __init__(self, beta_1, beta_2, steps, device):
        assert beta_1 < beta_2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        self.betas = torch.linspace(
            beta_1, beta_2, steps, dtype=torch.float32).to(device)
        self.sqrt_beta = torch.sqrt(self.betas)

        alphas = 1-self.betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        # 1/sqrt(alpha)
        self.oneover_sqrt_alpha = 1 / torch.sqrt(alphas)
        # sqrt(alpha_hat)
        self.alphas_hat_sqrt = torch.sqrt(alphas_hat)
        # sqrt(1-alpha_hat)
        self.sqrt_one_minus_alphas_hat = torch.sqrt(1 - alphas_hat)
        # (1 - alphas)/sqrt(1-alpha_hat)
        self.gamma = (1 - alphas) / (self.sqrt_one_minus_alphas_hat)

    def add_noise(self, x, step):
        sqrt_alpha_hat = self.alphas_hat_sqrt[step]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alphas_hat[step]
        sqrt_alpha_hat = sqrt_alpha_hat.reshape((x.shape[0], 1, 1, 1))
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.reshape(
            (x.shape[0], 1, 1, 1))
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        y = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return y, noise

    def denoise(self, Xt, step, pred_noise):
        z = torch.randn_like(Xt) if step > 1 else 0
    
        out = self.oneover_sqrt_alpha[step] * \
            (Xt - pred_noise * self.gamma[step]) + self.sqrt_beta[step] * z
        return out
