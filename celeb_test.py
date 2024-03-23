import os
import cv2
import time
import torch
import numpy as np

from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.basic_utils import Args
from model.model1 import LandmarkEstimationV1
from model.model2 import LandmarkEstimationV2


def eval(model, dataloader, args):
    model.eval()

    if not os.path.exists(f"{args.save_dir}/test"):
        os.makedirs(f"{args.save_dir}/test")

    nme_list= []
    cost_time = []
    with torch.no_grad():
        for idx, (img, _) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            img = img.to(args.device)
            model = model.to(args.device)

            start_time = time.time()
            _, predictions = model(img)
            cost_time.append(time.time() - start_time)

            predictions = predictions.cpu().numpy()
            predictions = predictions.reshape(predictions.shape[0], -1, 2)

            image = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
            image = (image * 255).astype(np.uint8)
            np.clip(image, 0, 255)
            
            image = np.ascontiguousarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_landmarks = predictions[0] * [112, 112]

            for (x, y) in pred_landmarks.astype(np.int32):
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

            cv2.imwrite(f"{args.save_dir}/test/{idx:>04}.jpg", image)
        print(f"inference_cost_time: {np.mean(cost_time):.4f}")


def main(args):
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.model_version == "v1":
        model = LandmarkEstimationV1().to(args.device)
    else:
        model = LandmarkEstimationV2().to(args.device)

    weights = torch.load(args.ckpt_dir)
    model.load_state_dict(weights)

    eval(model, dataloader, args)


if __name__ == "__main__":
    save_dir = "./runs/2024_03_22_18_02_44"
    data_dir = "/home/pervinco//Datasets/CelebA-HQ-256"

    args = Args(f"{save_dir}/config.yaml", is_train=False)
    args.save_dir = save_dir
    args.ckpt_dir = f"{save_dir}/ckpt/best.pth"
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)