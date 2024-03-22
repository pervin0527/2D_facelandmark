import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary
from model.model import LandmarkEstimationV1, LandmarkEstimationV2, AuxiliaryNet

if __name__ == "__main__":
    image_size = 112
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight_dir = "/home/pervinco/PFLD-pytorch/checkpoint/snapshot/checkpoint_epoch_500.pth.tar"
    model = LandmarkEstimationV2().to(device)
    summary(model, (3, image_size, image_size), batch_size=1, device="cuda")

    weights = torch.load(weight_dir)
    model.load_state_dict(weights["model"])