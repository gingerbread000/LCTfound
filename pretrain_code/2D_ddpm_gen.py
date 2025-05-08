import os
import time
import math
from PIL import Image

import torch
from transformers import BertTokenizer, BertModel

from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers import UNet2DModel, UNet2DConditionModel

from my_pipeline_ddpm import DDPMPipeline

@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 4
    eval_batch_size = 4 
    num_epochs = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    eval_freq_step = 2000
    mixed_precision = "fp16"  
    output_dir = "save_dir"
    num_workers = 4
    overwrite_output_dir = True  
    seed = 421
    pretrain_path = "./ddpm-lung-256-big-3/unet/"
    bert_path = "./chinesebert"

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def generate_seed():
    seed = (
        time.time_ns()
        ^ (os.getpid() << 16)
    )
    return seed

def evaluate(config, epoch, pipeline,prompt_embeding):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(generate_seed()),
        condition=prompt_embeding,
    ).images

    num_grid = int(math.sqrt(config.eval_batch_size))
    image_grid = make_grid(images, rows=num_grid, cols=num_grid)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return

def gen_loop(config, model, noise_scheduler):
    
    bert_path = config.bert_path
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("gen_example")

    model, bert_model = accelerator.prepare(model, bert_model)
    texts = [
        # 'window width and window level is 1500 and -500.',
        'window width and window level is 400 and 0.'
    ]
    cancer_texts = []
    
    for _ in range(1000):
        cancer_texts.append(
            texts[0] + \
            "this image may related to: 纵隔肿瘤" + ".")
    
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

    for i, text in enumerate(cancer_texts):
        prompt_embeding = []
        print(text)
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k:inputs[k].to(model.device) for k in inputs}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        prompt_embeding.append(cls_embedding.view(1, 1, -1))

        prompt_embeding = prompt_embeding * config.eval_batch_size
        prompt_embeding = torch.cat(prompt_embeding, dim=0)

        evaluate(config, i, pipeline, prompt_embeding)   
 

if __name__ == "__main__":
    config = TrainingConfig()

    #### Model setting ####
    model = UNet2DConditionModel(
        sample_size=config.image_size,  
        in_channels=3,  
        out_channels=3,  
        layers_per_block=2,  
        block_out_channels=(64, 128, 256, 512, 1024),
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
        encoder_hid_dim_type="text_proj",
        encoder_hid_dim=768,
    )

    model = model.from_pretrained(config.pretrain_path)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)  
    accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
    
    gen_loop(config, model, noise_scheduler)
