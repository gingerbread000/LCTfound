import os
import pdb
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

from util import get_dl, multi_acc, compute_iou, compute_dice, distillation_loss
from pixel_classifier import pixel_classifier
from fea_ex import FeatureExtractorDDPM, collect_features, prepare_fea
from unet import UNet


@dataclass
class TrainingConfig:
    image_size = 256  
    train_batch_size = 4 
    eval_batch_size = 4  
    num_epochs = 20
    gradient_accumulation_steps = 1
    learning_rate = 1e-3
    lr_warmup_steps = 100
    eval_freq_step = 2
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "save_dir" 
    upsample_mode = "bilinear"
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 123456
    model_number = 1
    mod = "pl"

config = TrainingConfig()
print(f"\033[31m {config.seed} \033[0m")
##########################################
seed = config.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##########################################


def evaluate_teacher(
        config, model, feature_extractor, accelerator, val_dataloader, epoch, global_step,
        ):
    progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    avg_acc = []
    avg_miou = []
    avg_2d_dice = []
    patient_metric = {}
    avg_3d_dice = []
    pid_flag = None
    patient_metric = {"pred":[], "inputs":[], "gt":[]}
    model.eval()
    feature_extractor.eval()
    for step, batch in enumerate(val_dataloader):
        clean_images = batch["image"]
        target = batch["label"].long().squeeze(1)
        bs = clean_images.shape[0]
        noise = torch.randn(clean_images.shape).to(clean_images.device)

        with torch.no_grad():
            out_fea, _, logits = feature_extractor(clean_images, noise)
            out_fea = collect_features(config, out_fea)
            out_fea, target = prepare_fea(out_fea, target)
            logits = model(out_fea)

            acc = multi_acc(logits, target)
            avg_acc.append(acc.item())
            miou = compute_iou(logits, target,bs)
            avg_miou.append(miou)
            dice = compute_dice(logits.detach().cpu(), target.detach().cpu(), bs)
            avg_2d_dice.append(dice)
            patient_ids = batch["id"]
            lenth = logits.shape[0] // bs
            for i, patient_id in enumerate(patient_ids):
                if pid_flag == None:
                    pid_flag = patient_id
                if patient_id == pid_flag:

                    patient_metric["pred"].append(logits.detach().cpu()[i*lenth:(i+1)*lenth])
                    patient_metric["gt"].append(target.detach().cpu()[i*lenth:(i+1)*lenth])
                    patient_metric["inputs"].append(clean_images.detach().cpu()[i*lenth:(i+1)*lenth])
                else:
                    d3_dice = compute_dice(
                        torch.cat(patient_metric["pred"], dim=0),
                        torch.cat(patient_metric["gt"], dim=0),
                        bs=1,
                        if_3d=True,
                    )
                    del(patient_metric)
                    patient_metric = {"pred":[], "inputs":[], "gt":[]}
                    avg_3d_dice.append(d3_dice)
                    pid_flag = patient_id

        progress_bar.update(1)
        logs = {"miou": np.mean(avg_miou), "acc":np.mean(avg_acc), "dice":np.mean(avg_2d_dice), "step": global_step}
        progress_bar.set_postfix(**logs)
    d3_dice = compute_dice(torch.cat(patient_metric["pred"], dim=0),torch.cat(patient_metric["gt"], dim=0),bs=1,if_3d=True,)
    avg_3d_dice.append(d3_dice)
    
    return np.mean(avg_3d_dice)

def evaluate(
        config, model, accelerator, val_dataloader, epoch, global_step,
        ):
    progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    avg_acc = []
    avg_miou = []
    avg_2d_dice = []
    patient_metric = {}
    model.eval()
    for step, batch in enumerate(val_dataloader):
        clean_images = batch["image"]
        target = batch["label"].long().squeeze(1)
        bs = clean_images.shape[0]

        with torch.no_grad():
            logits = model(clean_images)

            acc = multi_acc(logits, target)
            avg_acc.append(acc.item())
            miou = compute_iou(logits, target,bs)
            avg_miou.append(miou)
            dice = compute_dice(logits.detach().cpu(), target.detach().cpu(), bs)
            avg_2d_dice.append(dice)
            patient_ids = batch["id"]
            lenth = logits.shape[0] // bs
            for i, patient_id in enumerate(patient_ids):
                if patient_id not in patient_metric:
                    patient_metric[patient_id] = {"pred":[], "gt":[], "inputs":[]}
                patient_metric[patient_id]["pred"].append(logits.detach().cpu()[i*lenth:(i+1)*lenth])
                patient_metric[patient_id]["gt"].append(target.detach().cpu()[i*lenth:(i+1)*lenth])
                patient_metric[patient_id]["inputs"].append(clean_images.detach().cpu()[i*lenth:(i+1)*lenth])

        progress_bar.update(1)
        logs = {"miou": np.mean(avg_miou), "acc":np.mean(avg_acc), "dice":np.mean(avg_2d_dice), "step": global_step}
        progress_bar.set_postfix(**logs)
    avg_3d_dice = []
    for patient_id in patient_metric:
        d3_dice = compute_dice(
                    torch.cat(patient_metric[patient_id]["pred"], dim=0),
                    torch.cat(patient_metric[patient_id]["gt"], dim=0),
                    bs=1,
                    if_3d=True,
                    )
        avg_3d_dice.append(d3_dice)

    return np.mean(avg_3d_dice)

def train_loop(config, teacher, student, accelerator, feature_extractor, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler, model_id):
    
    train_dataloader, val_dataloader, test_dataloader, teacher, student, feature_extractor, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, test_dataloader, teacher, student, feature_extractor, optimizer, lr_scheduler
    )

    global_step = 0
    best_score = 0
    print("start training.")
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
        progress_bar.set_description(f"Epoch {epoch}")

        avg_acc = []
        avg_miou = []
        avg_2d_dice = []
        teacher.eval()
        feature_extractor.eval()
        student.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["image"]
            target = batch["label"].long().squeeze(1)

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            with accelerator.accumulate(student):
                
                out_fea, _, logits = feature_extractor(clean_images, noise)
                out_fea = collect_features(config, out_fea)
                out_fea, target1 = prepare_fea(out_fea, target)
                logits_teacher = teacher(out_fea)
                logits_teacher = logits_teacher.view(
                    config.train_batch_size, 
                    config.image_size,
                    config.image_size,
                    -1
                    )
                logits_teacher = logits_teacher.permute(0,3,1,2)
                logits_student = student(clean_images)
                kl_loss = distillation_loss(logits_student, logits_teacher)
                target1 = target < 0
                target[target1] = -1
                loss = F.cross_entropy(logits, target=target, ignore_index=-1)
                ce_loss = F.cross_entropy(logits, target, reduction='none',ignore_index=-1)
                pt = torch.exp(-ce_loss)
                alpha = 0.25
                gamma = 2
                focal_loss = ( alpha * (1-pt)** gamma * ce_loss).mean()
                loss += focal_loss + kl_loss
                
                acc = multi_acc(logits_student, target)
                avg_acc.append(acc.item())

                miou = compute_iou(logits_student, target, bs)
                avg_miou.append(miou)
                dice = compute_dice(logits_student.detach().cpu(), target.detach().cpu(), bs)
                avg_2d_dice.append(dice)

                accelerator.backward(loss)

                accelerator.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "miou":np.mean(avg_miou),
                "acc":np.mean(avg_acc),
                "dice":np.mean(avg_2d_dice),
                "step": global_step
                }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if (epoch + 1) % config.eval_freq_step == 0:

            avg_3d_dice = evaluate(
                config, student, accelerator, val_dataloader, epoch, global_step,
                )
            avg_3d_dice = accelerator.gather(torch.tensor(avg_3d_dice).cuda()).cpu().numpy()
            avg_3d_dice = np.mean(avg_3d_dice)
            print('Student val dice:', avg_3d_dice)

            if avg_3d_dice > best_score and accelerator.is_main_process:
                best_score = avg_3d_dice
                log_best = {"epoch":epoch, "global step": global_step, "train_iou":np.mean(avg_miou), "val_3d_dice":best_score}
                torch.save(
                    {
                        "student": accelerator.unwrap_model(student).state_dict(), 
                    },
                    os.path.join(config.output_dir, f"unet_{model_id}.pth"))
                print("new best val mertic: ", log_best)
            if epoch == (config.num_epochs - 1):
                torch.save(
                    {
                        "student": accelerator.unwrap_model(student).state_dict(), 
                    },
                    os.path.join(config.output_dir, f"unet_last.pth"))
                
    if accelerator.is_main_process:
        print(log_best)

if __name__ == "__main__":


    accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                log_with="tensorboard",
                project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    train_dataloader, val_dataloader, test_dataloader = get_dl(config)
    for i in range(config.model_number):
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(
            model_path="/path/to/pretrained/ddpm/model",
            input_activations=False,
            steps=[0, 50, 100, 150],
            blocks=[
                [],
                [1, 2, 3, 4],
            ],
            )
        for paras in feature_extractor.parameters():
            paras.requires_grad = False

        teacher = pixel_classifier(2, 960*4)
        checkpoint = "/path/to/large_ddpm_seg/model"
        ckpt = torch.load(checkpoint, map_location="cpu")
        feature_extractor.load_state_dict(ckpt["backbone"])
        teacher.load_state_dict(ckpt["mlp"])
        
        for paras in teacher.parameters():
            paras.requires_grad = False

        student = UNet(3,2)
        
        optimizer = torch.optim.AdamW(
            [
                {"params":student.parameters()}
                ],
            lr=config.learning_rate,
            weight_decay=0.01,
            )
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        train_loop(
            config, teacher, student, accelerator, feature_extractor, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler, i
        )
