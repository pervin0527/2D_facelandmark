import os
import cv2
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.basic_utils import Args
from data.dataset import WFLWDataset
from model.model1 import LandmarkEstimationV1
from model.model2 import LandmarkEstimationV2

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def eval(model, dataloader, args):
    model.eval()

    if not os.path.exists(f"{args.save_dir}/test"):
        os.makedirs(f"{args.save_dir}/test")

    nme_list= []
    cost_time = []
    with torch.no_grad():
        for idx, (img, landmark_gt, _, _) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            img = img.to(args.device)
            landmark_gt = landmark_gt.to(args.device)
            model = model.to(args.device)

            start_time = time.time()
            _, predictions = model(img)
            cost_time.append(time.time() - start_time)

            predictions = predictions.cpu().numpy()
            predictions = predictions.reshape(predictions.shape[0], -1, 2)

            gt_landmarks = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()

            image = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
            image = (image * 255).astype(np.uint8)
            np.clip(image, 0, 255)
            
            image = np.ascontiguousarray(image)
            pred_landmarks = predictions[0] * [112, 112]

            for (x, y) in pred_landmarks.astype(np.int32):
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
            cv2.imwrite(f"{args.save_dir}/test/{idx:>04}.jpg", image)
            
            nme_temp = compute_nme(predictions, gt_landmarks)
            for item in nme_temp:
                nme_list.append(item)

        print(f'nme: {np.mean(nme_list):.4}')
        print(f"inference_cost_time: {np.mean(cost_time):.4f}")


def main(args):
    test_file = f"{args.data_dir}/test/list.txt"
    test_dataset = WFLWDataset(test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.model_version == "v1":
        model = LandmarkEstimationV1().to(args.device)
    else:
        model = LandmarkEstimationV2().to(args.device)

    ckpt = torch.load(args.ckpt_dir)
    model.load_state_dict(ckpt)
    
    eval(model, test_dataloader, args)

if __name__ == "__main__":
    save_dir = "./runs/2024_03_22_18_02_44"
    args = Args(f"{save_dir}/config.yaml", is_train=False)

    args.save_dir = save_dir
    args.ckpt_dir = f"{save_dir}/ckpt/best.pth"
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)