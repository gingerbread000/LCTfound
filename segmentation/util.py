import os
import cv2
import json
import natsort
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monai import transforms
from monai.transforms import apply_transform

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    if len(y_pred_tags.shape) == 3:
        b, h, w = y_pred_tags.shape
        total_pixel = b * h * w
    else:
        total_pixel = len(y_pred_tags)
    acc = correct_pred.sum() / total_pixel

    acc = acc * 100

    return acc

def compute_iou(preds, gts, bs, target_num=1):

    gts = gts.detach().cpu().numpy()

    y_pred_softmax = torch.log_softmax(preds, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    y_pred_tags = y_pred_tags.detach().cpu().numpy()
    if target_num == -1:
        preds_tmp = (y_pred_tags > 0).astype(int)
        gts_tmp = (gts > 0).astype(int)
        target_num = 1
    else:
        preds_tmp = (y_pred_tags == target_num).astype(int)
        gts_tmp = (gts == target_num).astype(int)
    ious = []
    lenth = len(preds) // bs
    for i in range(bs):
        sub_preds_tmp = preds_tmp[i*lenth:(i+1)*lenth]
        sub_gts_tmp = gts_tmp[i*lenth:(i+1)*lenth]

        unions = Counter()
        intersections = Counter()
        unions[target_num] += (sub_preds_tmp | sub_gts_tmp).sum()
        intersections[target_num] += (sub_preds_tmp & sub_gts_tmp).sum()

        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)

    return np.array(ious).mean()

def compute_dice(preds, gts, bs, if_3d=False, target_num=1):
    if if_3d:
        bs = 1
    
    gts = gts.numpy()

    y_pred_softmax = torch.log_softmax(preds, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    y_pred_tags = y_pred_tags.numpy()
    if target_num == -1:
        preds_tmp = (y_pred_tags > 0).astype(int)
        gts_tmp = (gts > 0).astype(int)
        target_num = 1
    else:
        preds_tmp = (y_pred_tags == target_num).astype(int)
        gts_tmp = (gts == target_num).astype(int)
    lenth = len(preds) // bs
    dices = []
    for i in range(bs):
        sub_preds_tmp = preds_tmp[i*lenth:(i+1)*lenth]
        sub_gts_tmp = gts_tmp[i*lenth:(i+1)*lenth]

        dice = (2 * (sub_gts_tmp * sub_preds_tmp).sum() / (1e-8 + sub_gts_tmp.sum() + sub_preds_tmp.sum() ))
        dices.append(dice)
    return np.array(dices).mean()

class MDataset(Dataset):
    def __init__(
        self,
        image_size,
        transform,
        section = "train",
        use_dist=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.section = section
        with open("split_final.json", "r") as f:
            split = json.load(f)
        self.paths = []
        if section in split:
            select_pid = list(split[section].keys())
            for k in select_pid:
                for f in split[section][k]:
                    if os.path.exists(os.path.join(f"/path/to/data/{section}/", f)):
                        self.paths.append(
                            os.path.join(f"/path/to/data/{section}", f)
                        )
        self.gen_img = []
        if section == 'train':
            if use_dist:
                gen_fp = 'generated_data/1'
                self.gen_img = [os.path.join(gen_fp, fn) for fn in os.listdir(gen_fp)]
                self.gen_img = self.gen_img[:16000]
            else:
                self.gen_img = []

        if section != "train":
            self.paths = natsort.natsorted(self.paths)
        
        self.transform = transform
        print("file number : ", self.__len__())

    def __len__(self):
        if self.section == 'train':
            if self.gen_img != []:
                return len(self.paths) + len(self.gen_img)
        return len(self.paths)

    def __getitem__(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        if self.section == 'train' and self.gen_img != [] and index > len(self.paths)-1:
            fp = self.gen_img[index - (len(self.paths)-1)-1]
            data_i = np.array(Image.open(str(fp)))
            label = np.zeros((data_i.shape[0], data_i.shape[1])) - 1
        
            cv2.setNumThreads(0)
            data_i = data_i / 255.
            data_i = np.transpose(data_i, (2,0,1))
            label = label[None, :, :]
            patient_id = str(fp).split("/")[-1]
        else:
            fp = self.paths[index]
            data_i = np.array(Image.open(str(fp).replace("_seg.png", ".png")))
            label = np.array(Image.open(fp))
        
            cv2.setNumThreads(0)
            data_i = cv2.cvtColor(data_i, cv2.COLOR_GRAY2RGB) / 255.
            data_i = np.transpose(data_i, (2,0,1))
            label = label[None, :, :]

            patient_id = str(fp).split("/")[-1]
            patient_id = "_".join(patient_id.split("_")[:-2])
        data = {"image":data_i, "label":label, "report": "nothing.", "id": patient_id}
        del(data_i)
        del(label)
        return apply_transform(self.transform, data) if self.transform is not None else data

def get_dl(config):
    train_transforms = transforms.Compose(
        [
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.Resized(keys=["image", "label"], spatial_size=[config.image_size,config.image_size]),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandRotated(keys=["image", "label"], range_x=0.2, range_y=0.2, prob=0.5),
            transforms.ToTensord(keys=["image", "report"],allow_missing_keys=True),
            transforms.NormalizeIntensityd(keys=["image"], subtrahend=0.5, divisor=0.5),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.Resized(keys=["image", "label"], spatial_size=[config.image_size,config.image_size]),
            transforms.ToTensord(keys=["image", "report"],allow_missing_keys=True),
            transforms.NormalizeIntensityd(keys=["image"], subtrahend=0.5, divisor=0.5),
        ]
    )

    dataset = MDataset(config.image_size, train_transforms, section="train", mod=config.mod)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=8)

    vdataset = MDataset( config.image_size, val_transforms, section="val", mod=config.mod)
    val_dataloader = DataLoader(vdataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=True, num_workers=8)

    tdataset = MDataset(config.image_size, val_transforms, section="test", mod=config.mod)
    test_dataloader = DataLoader(tdataset, batch_size=config.eval_batch_size, shuffle=False, drop_last=True, num_workers=8)
    return train_dataloader, val_dataloader, test_dataloader


def distillation_loss(student_logits, teacher_logits, T=3):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(soft_student, soft_teacher, reduction='mean') * (T**2)

class DistillationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, T=3):
        super().__init__()
        self.alpha = alpha  
        self.beta = beta  
        self.T = T

    def forward(self, student_logits, teacher_logits, y_true):
        ce_loss = F.cross_entropy(student_logits, y_true)
        task_loss = ce_loss

        kld_loss = distillation_loss(student_logits, teacher_logits, self.T)

        # 总损失
        total_loss = self.alpha * task_loss + self.beta * kld_loss
        return total_loss