import torch
from torch import nn

class LandmarkCost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize, device):
        ## 예측된 오일러 각도와 실제 오일러 각도 사이의 차이 계산.
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        
        ## 속성 가중치 계산.
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        mat_ratio = torch.Tensor([1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio]).to(device)
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)

        ## L2 Distance를 통해 landmark간 차이 계산.
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)