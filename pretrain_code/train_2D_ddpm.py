import os
import pdb
import math
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from accelerate import Accelerator

from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers import DDPMScheduler

from my_pipeline_ddpm import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.util import get_dl
import safetensors
from transformers import BertTokenizer, BertModel

@dataclass
class TrainingConfig:
    image_size = 256  
    train_batch_size = 24
    eval_batch_size = 16
    num_epochs = 6
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 10000
    eval_freq_step = 10000
    mixed_precision = "bf16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "save_dir"  # the model name locally and on the HF Hub
    num_workers = 4
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 421
    bert_path = "./chinesebert"

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline,prompt_embeding):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        condition=prompt_embeding,
    ).images

    # Make a grid out of the images
    num_grid = int(math.sqrt(config.eval_batch_size))
    image_grid = make_grid(images, rows=num_grid, cols=num_grid)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config, accelerator, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    bert_path = config.bert_path
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)
    
    # Initialize accelerator and tensorboard logging
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler, bert_model = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, bert_model
    )
    global_step = 0
    print("start training.")
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):

            prompt_embeding = []
            for text in batch[1]:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                inputs = {k:inputs[k].to(batch[0].device) for k in inputs}
                with torch.no_grad():
                    outputs = bert_model(**inputs)
            
                # 获取最后一层的隐藏状态，取 [CLS] token 的向量作为句子嵌入
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                prompt_embeding.append(cls_embedding.view(1, 1, -1))
            prompt_embeding = torch.cat(prompt_embeding, dim=0)
            # pdb.set_trace()
            clean_images = batch[0][:,:3,:,:]

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=prompt_embeding, return_dict=False, )[0]
                
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process and global_step % config.eval_freq_step == 0:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # if (epoch + 1) % config.eval_freq_step == 0 or (step + 1) == len(train_dataloader):
                evaluate(config, epoch*len(train_dataloader)+step, pipeline, prompt_embeding)
                # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                # if  (step + 1) == len(train_dataloader):
                pipeline.save_pretrained(config.output_dir)

if __name__ == "__main__":
    config = TrainingConfig()


    #### dataset setting and dataloader setting ####
    train_dataloader, train_dataset = get_dl(config=config)

    # fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # for i, image in enumerate(train_dataset):
    #     if i == 4: break
    #     image = np.transpose(image, (1,2,0))
    #     image = (image - image.min()) / (image.max() - image.min())
    #     axs[i].imshow(image)
    #     axs[i].set_axis_off()
    # fig.show()
    # plt.savefig("samples.png")
    # plt.close()

    #### Model setting ####
    model = UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 256, 512, 1024),  # the number of output channels for each UNet block
        # block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D",  
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        norm_num_groups=32,
        # addition_embed_type="text",
        # addition_embed_type_num_heads=64,
        encoder_hid_dim_type="text_proj",
        encoder_hid_dim=768,
    )

    model_path = "/GPUFS/gyfyy_jxhe_1/User/gaozebin/project/ddpm/2d_ddpm_lung_with_ca/ddpm-lung-256-big-1/unet"
    pretrained_w = safetensors.torch.load_file(model_path + "/diffusion_pytorch_model.safetensors")
    model.load_state_dict(pretrained_w, strict=False)
    
    sample_image, _ = train_dataset[10]
    sample_image = sample_image.unsqueeze(0)
    print("Input shape:", sample_image.shape)
    # out = model(sample_image, timestep=0).sample
    # print("Output shape:", out.shape)
    # print("Out value:", out.min(), out.max())

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([50])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]).save("noise_img.png")

    # noise_pred = model(noisy_image, timesteps).sample
    # loss = F.mse_loss(noise_pred, noise)
    # print(loss)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
    
    train_loop(config, accelerator, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
