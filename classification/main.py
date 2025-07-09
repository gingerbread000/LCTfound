# License: BSD
# Author: Sasank Chilamkurthy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from data_utils import MDataset_2D
import time
import os
import pickle
from dataclasses import dataclass
import numpy as np
import time

from UNet2D import CLSDDPM_2D
from dataclasses import dataclass
from accelerate import Accelerator

cudnn.benchmark = True
import argparse
from monai.metrics.rocauc import compute_roc_auc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1234567, help="Random seed to use.")
    parser.add_argument("--backbone", type=str, default="ddpm")
    parser.add_argument("--experiment", type=str, default="some",help="Mlflow experiment name.")

    args = parser.parse_args()
    return args
args = parse_args()

def set_seed(args):
    ##########################################
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ##########################################
set_seed(args)

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    batch_size = 2
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-3
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "save_dir"  # the model name locally and on the HF Hub

    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    num_workers = 4

config = TrainingConfig()

batch_size = config.batch_size
image_size = config.image_size
tr_dataset = MDataset_2D(image_size,section='train')
val_dataset = MDataset_2D(image_size,section='val')
test_dataset = MDataset_2D(image_size,section='test')

image_datasets = {"train": tr_dataset, "val": val_dataset, "test": test_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=x=="train", num_workers=config.num_workers,pin_memory=False)
              for x in ['train', 'val', "test"]}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = ["2", "greater than 2"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(dataset_sizes)


def train_model(config, model, dataloaders, criterion, optimizer, scheduler, num_epochs=15, mode="Train"):
    since = time.time()
    phase_candidate = ['train', 'val', ]

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    training_dataloader, val_dataloader, test_loader = dataloaders["train"], dataloaders["val"], dataloaders["test"]
    model, optimizer, training_dataloader, val_dataloader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, val_dataloader, test_loader, scheduler
        )
    
    if mode == "Test":
        saved = torch.load(f"checkpoints/saved.pkl")
        saved = {k.replace("module.", ""):saved[k] for k in saved}
        model.load_state_dict(saved)
        num_epochs = 1
        phase_candidate = ['val'] # or 'test'

    dataloaders = {"train":training_dataloader, "val":val_dataloader, "test": test_loader}
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        set_seed(args)
        if accelerator.is_main_process: print(f'Epoch {epoch}/{num_epochs - 1}')
        if accelerator.is_main_process: print('-' * 10)

        for phase in phase_candidate:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            epoch_p = {}
            epoch_t = {}
            iid = {}
            iifn = {}
            step = 1
            c = 0
            print(phase)
            st = time.time()
            for inputs, labels, id_list, fn_list, _ in dataloaders[phase]:
                labels = labels.float()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.flatten(), labels)

                    if phase == 'train':
                        accelerator.backward(loss)
                        optimizer.step()
                logits = torch.sigmoid(outputs).detach().cpu()
                labels = (labels > 0.5).float()
                for i in range(inputs.shape[0]):
                    
                    epoch_p[c] = logits[i][None]
                    epoch_t[c] = labels[i].detach().cpu()[None]
                    iid[c] = id_list[i]
                    iifn[c] = fn_list[i]
                    c += 1
                if step % 400 == 0 and accelerator.is_main_process:
                    print(f"Step {step} loss: {running_loss / (inputs.size(0) * step)}")
                    print(np.sum(labels.detach().cpu().numpy()) / len(labels))
                    print(time.time() - st)
                    st = time.time()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((logits.data > 0.5) == labels.detach().cpu().data)
                step += 1

            patient_p = {}
            patient_t = {}
            for i,k  in enumerate(epoch_p):
                if iid[k] not in patient_p:
                    patient_p[iid[k]] = []
                    patient_t[iid[k]] = []
                patient_p[iid[k]].append(epoch_p[k])
                patient_t[iid[k]].append(epoch_t[k])
            patient_p = [torch.mean(torch.cat(v)) for k,v in patient_p.items()]
            patient_t = [torch.mean(torch.cat(v)) for k,v in patient_t.items()]
            patient_auc =  compute_roc_auc(torch.tensor(patient_p), torch.tensor(patient_t))

            epoch_p = accelerator.gather(torch.cat(list(epoch_p.values())))
            epoch_t = accelerator.gather(torch.cat(list(epoch_t.values())))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (dataset_sizes[phase])
            if accelerator.is_main_process:
                aucs = []
                auc = compute_roc_auc(epoch_p, epoch_t)
                
                aucs.append(auc)
                auc = np.mean([i for i in aucs if i != 0])
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Auc: {auc:.4f} PAUC: {patient_auc:.4f}')
                print([round(i,4) for i in aucs])

                if phase == 'val':
                    if auc > best_auc:
                        best_auc = auc
                        with open(f"res/{args.backbone}_{args.experiment}_interval_best.pkl", "wb") as f:
                            pickle.dump({"pred": epoch_p, "gt": epoch_t, "id":iid, "fn":iifn},f)
                        if mode == "Train":
                            model.eval()
                            torch.save(model.state_dict(), f"checkpoints/{args.backbone}_best_{args.experiment}_{np.round(auc,3)}.pkl")
                    
                    if mode == "Train":
                        torch.save(model.state_dict(), f"checkpoints/{args.backbone}_{args.experiment}_last.pkl")
                    
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Auc {args.experiment}: {best_auc:4f} {args.seed}_{args.backbone}')
    
    return model

backbone_name = args.backbone
mode = "Train"
num_cls = 1

model_ft = CLSDDPM_2D(
    "/path/to/pretrained/ddpm/model",  # Path to the pretrained DDPM model
    num_cls=num_cls,
    seq_len=3,
    steps=[1,20,50],
    )
for paras in model_ft.model.parameters():
    paras.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, weight_decay=1e-3) # or SGD

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
model_ft = train_model(config, model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=4, num_cls=num_cls, mode=mode)
