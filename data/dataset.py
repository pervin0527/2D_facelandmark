import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset

class WFLWDataset(Dataset):
    def __init__(self, txt_file):
        with open(txt_file, "r") as f:
            self.lines = f.readlines()
            
        self.transforms = self.get_basic_transform()

    def __getitem__(self, idx):
        self.line = self.lines[idx].strip().split()
        
        self.image = cv2.imread(self.line[0])
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32)

        self.image = self.transforms(self.image)
        return (self.image, self.landmark, self.attribute, self.euler_angle)

    def __len__(self):
        return len(self.lines)
    
    def get_basic_transform(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])

        return transform