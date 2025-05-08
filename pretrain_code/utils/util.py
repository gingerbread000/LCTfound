import os
import cv2
import json
import pickle
import numpy as np

from torch.utils.data import DataLoader, Dataset

from monai import transforms
import json

def save_best_preds(mdict, path):
    with open(path, "wb") as f:
        pickle.dump(mdict, f)

class MDataset(Dataset):
    def __init__(self, section=True, transform=None, prompt_embeding=None) -> None:
        self.section = section
        self.transform = transform
        self.paths = []
        with open("../total_file_path.pkl", "rb") as f:
            self.paths = pickle.load(f)
        with open('pid_map_info.json', 'r') as f:
            self.iid_map_info = json.load(f)
        
        print(len(self.paths)) 

    def __len__(self):
        return len(self.paths)
    
    def read_npy(self, fp):
        img = np.load(fp)["im"]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) / 255.
        img = np.transpose(img, (2,0,1))
    
        return img

    def _transform(self, fp):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.read_npy(fp)
        return self.transform(data_i)

    def __getitem__(self, index):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        text = ''
        fp = self.paths[index]
        fn = os.path.basename(fp)
        if np.random.random() > 0.05:
            if fn.startswith('0'):
                text += 'window width and window level is 1500 and -500.'
            else:
                text += 'window width and window level is 400 and 0.'
            if np.random.random() > 0.5:
                iid = os.path.basename(os.path.dirname(fp))
                if iid in self.iid_map_info:
                    text += "this image may related to " + self.iid_map_info[iid]
        else:
            text = 'freedom.'

        return self._transform(fp), text
    
    
def get_dl(config):
    train_transforms = transforms.Compose([
            transforms.RandScaleCrop([0.9,0.9],[1.1,1.1],random_size=True),
            transforms.Resize([config.image_size,config.image_size]),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandRotate90(prob=0.3),
            transforms.ToTensor(),
            transforms.RandAdjustContrast(prob=0.1, gamma=(0.97, 1.03)),
            transforms.NormalizeIntensity(0.5, 0.5),
        ])

    dataset = MDataset(section="train",transform=train_transforms)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)

    return train_dataloader, dataset
