import json
import natsort
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from monai import transforms
from torch.utils.data import Dataset


class MDataset_2D(Dataset):
    def __init__(
        self,
        image_size,
        section="train",
        processor=None,
        use_processor=False
    ):
        super().__init__()
        fp = "/path/to/data/"
        with open("/path/to/split.json", "r") as f:
            self.split = json.load(f)
        self.labels = np.load("/path/to/label.npy", allow_pickle=True).item()
        pids = self.split[section]
        
        self.clinic_embed = {
            'female':np.array([0]), 'male':np.array([1]), 'left':np.array([0]), 'right':np.array([1]), 
            'old':np.array([0]), 'young':np.array([1]), 'standard':np.array([0]), 'trachea':np.array([0.5]),'obesity':np.array([1]), 
            }
        
        self.paths = []
        
        iids = []
        for iid in pids:
            cts = glob(f"{fp}/{iid}*.png")
            if cts != []:
                iids.append(iid)
                cts = natsort.natsorted(cts)
                if section == "train":
                    self.paths += cts[len(cts) // 2 - 6 : len(cts) // 2 + 6]
                else:
                    self.paths += cts[len(cts) // 2 : len(cts) // 2 +1]
        print(section, iids)
        self.transform = transforms.Compose([
            transforms.Resize([image_size,image_size]) ,
            transforms.RandRotate(range_x=0.5, range_y=0.5, prob=0.5) if section == "train" else nn.Identity(),
            transforms.ToTensor(),
            transforms.RandCoarseDropout(4, (16,16)) if section == "train" else nn.Identity(),
        ])
        self.use_processor = use_processor
        if use_processor:
            self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        fp = self.paths[index]
        fn = fp.split("/")[-1]
        pid = fn.split("_")[0]
        label = self.labels[pid]

        arr_f = np.array(Image.open(fp).convert("L")) / 255.
        arr_zg = np.array(Image.open(fp.replace(".png", "_zg.png")).convert("L")) / 255.

        embed_vec = []
        for word in self.clinic[pid]:
            vec = torch.tensor(self.clinic_embed[word]).float()
            embed_vec.append(vec)

        embed_vec = torch.cat(embed_vec, dim=-1)
        embed_vec = embed_vec.squeeze(0)

        arr = np.zeros((3,self.image_size, self.image_size))
        arr[0] = arr_f
        arr[1] = arr_f
        arr[2] = arr_f

        if self.use_processor:
            res = self.processor.image_processor(
                images=self.transform(arr), do_rescale=False
                )
            
            return res, label, pid, fn, embed_vec
        
        return self.transform(arr), label, pid, fn, embed_vec
