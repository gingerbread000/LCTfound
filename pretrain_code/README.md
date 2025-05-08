# LCTfound-Pretrain: Diffusion Model Pretraining for Medical Image Generation

This repository contains code for **pretraining diffusion models** to generate lung CT images, guided by text prompts using `ChineseBERT`.

> 🧠 **ChineseBERT** is used to tokenize and embed textual prompts that condition the image generation process.

---

## 📦 Project Structure

- `train_2D_ddpm.py`: Main script for training.
- `2D_ddpm_gen.py`: Main script for generation.
- `my_pipeline_ddpm.py`: Customized diffusion pipeline.
- `single.yaml`: Accelerator configuration file.

---

## 🧪 Pretraining Instructions

To start pretraining the model, ensure you have properly set the path to your pretrained **ChineseBERT** model in the config.

```bash
cd pretrain_code
accelerate launch \
  --gpu_ids=0 \
  --num_processes=1 \
  --main_process_port=6676 \
  --config_file=single.yaml \
  train_2D_ddpm.py
```

---

## 🖼️ Image Generation

Once pretrained, you can generate images using the same script and configuration:

```bash
cd pretrain_code
accelerate launch \
  --gpu_ids=0 \
  --num_processes=1 \
  --main_process_port=6676 \
  --config_file=single.yaml \
  2D_ddpm_gen.py
```
---

## 🔗 Pretrained Weights

Pretrained model weights can be downloaded from the following link: [**Download Here**](https://drive.google.com/file/d/1AGAWtMMErr2jJEjdAfSu78h2ZOB6-zww/view?usp=drive_link)

---

## 🔧 Requirements

Make sure the following packages are installed:

* `transformers`
* `torch`
* `diffusers`
* `accelerate`
* `Pillow`

---

## 📁 Notes

* Ensure `ChineseBERT` is available at the path specified in your config (e.g., `./chinesebert`).
* Images are saved in `save_dir/samples/` during generation.


