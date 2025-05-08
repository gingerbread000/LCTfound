# 🫁 Zero-Shot Super-Resolution on Lung CT using LCTfound

This project explores **zero-shot lung CT super-resolution** based on the pre-trained **LCTfound** foundation model. The goal is to enhance low-resolution CT scans without any task-specific fine-tuning, leveraging the generative power and semantic priors of a diffusion-based backbone.

We extend the capabilities of LCTfound by integrating a **zero-shot image restoration pipeline**, which aligns with recent advancements in generative diffusion models.


## 🧠 Acknowledgements

We thank the authors of the excellent work:

> **Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model**  
> [https://github.com/wyhuai/DDNM](https://github.com/wyhuai/DDNM)  

## 📂 Code Structure

- `script_Lung_CT.py`: Main script for zero-shot inference on lung CT super-resolution using LCTfound.

