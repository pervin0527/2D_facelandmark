import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler

def visualize_inference(model, auxiliary_net, dataloader, device, save_path, epoch):
    model.eval()
    auxiliary_net.eval()

    dataiter = iter(dataloader)
    images, landmark_gt, _, _ = next(dataiter)
    image = images[0].unsqueeze(0).to(device)

    with torch.no_grad():
        _, pred_landmark = model(image)

    pred_landmark = pred_landmark.cpu().numpy().reshape(-1, 2)  # landmark
    image = np.transpose(image[0].cpu().numpy(), (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    image = np.clip(image, 0, 255)
    image = np.ascontiguousarray(image)

    pred_landmark = pred_landmark * [image.shape[1], image.shape[0]]
    for (x, y) in pred_landmark.astype(np.int32):
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"{save_path}/imgs/{epoch:>04}.png")
    plt.close()


class WarmUpCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, max_lr=1e-3, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.end_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # WarmUp Phase
            lr = ((self.max_lr - self.min_lr) / self.warmup_epochs) * self.last_epoch + self.min_lr
        else:
            # Cosine Annealing Phase
            cosine_decay = 0.5 * (1 + torch.cos(torch.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            lr = (self.max_lr - self.end_lr) * cosine_decay + self.end_lr
        
        return [lr for group in self.optimizer.param_groups]