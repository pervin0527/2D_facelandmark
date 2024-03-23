import os
import torch

from tqdm import tqdm
from torchsummary import summary
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils.basic_utils import Args
from utils.train_utils import visualize_inference, WarmUpCosineScheduler

from loss import LandmarkCost
from data.dataset import WFLWDataset
from model.model1 import LandmarkEstimationV1, AuxiliaryNetV1
from model.model2 import LandmarkEstimationV2, AuxiliaryNetV2


def train(dataloader, model, auxiliary, criterion, optimizer, epoch, device):
    model.train()
    auxiliary.train()
    total_loss = 0.0
    total_count = 0

    for image, landmark_gt, attribute_gt, euler_angle_gt in tqdm(dataloader, desc="Training", leave=False):
        bs = image.size(0)
        image = image.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)

        features, landmarks = model(image)
        angle = auxiliary(features)

        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, bs, device)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss += loss.item() * bs
        total_count += bs

    avg_loss = total_loss / total_count
    return avg_loss


def eval(dataloader, model, auxiliary, criterion, device):
    model.eval()
    auxiliary.eval()
    total_loss = 0.0
    total_count = 0 

    with torch.no_grad():
        for image, landmark_gt, attribute_gt, euler_angle_gt in tqdm(dataloader, desc="Evaluating", leave=False):
            bs = image.size(0)
            image = image.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)

            features, landmarks = model(image)
            angle = auxiliary(features)

            weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, bs, device)

            total_loss += loss.item() * bs
            total_count += bs

    avg_loss = total_loss / total_count
    return avg_loss


def main(args):
    writer = SummaryWriter(log_dir=f"{args.save_dir}/logs")

    train_file = f"{args.data_dir}/train/list.txt"
    test_file = f"{args.data_dir}/test/list.txt"

    train_dataset = WFLWDataset(train_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    test_dataset = WFLWDataset(test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.model_version == "v1":
        model = LandmarkEstimationV1().to(args.device)
        auxiliary_net = AuxiliaryNetV1().to(args.device)
    else:
        model = LandmarkEstimationV2().to(args.device)
        auxiliary_net = AuxiliaryNetV2().to(args.device)
    
    if args.build_test:
        summary(model, (3, args.image_size, args.image_size), batch_size=1, device="cuda")

    criterion = LandmarkCost()
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': auxiliary_net.parameters()}],
                                 lr=args.init_lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr)

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch : {epoch}|{args.epochs}, LR : {current_lr:.6f}")
        train_loss = train(train_dataloader, model, auxiliary_net, criterion, optimizer, epoch, args.device)
        valid_loss = eval(test_dataloader, model, auxiliary_net, criterion, args.device)
        print(f"Train Loss {train_loss:.4f}, Valid Loss {valid_loss:.4f}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{args.save_dir}/ckpt/best.pth")
            print(f"Valid Loss improved, saving model.")

        visualize_inference(model, auxiliary_net, test_dataloader, args.device, args.save_dir, epoch)
        # scheduler.step(valid_loss)
        scheduler.step()

    writer.close()


if __name__ == "__main__":
    args = Args("./hyps.yaml", is_train=True)
    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)