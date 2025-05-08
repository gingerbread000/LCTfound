import os
import io
import glob
import time
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as tvu
import torchvision.models as models

from PIL import Image
from scipy.linalg import orth
from skimage.util import random_noise
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
import tifffile

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
# from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

from guided_diffusion.models import Model
from guided_diffusion.script_util import (
    create_model,
    create_classifier,
    classifier_defaults,
    args_to_dict,
    create_model_lung_ct_v1,
)


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def get_poisson_noisy_img(img, lamb=0.1):
    """
    Add Poisson noise to the image.

    Args:
        image (torch.tensor): Input image.

    Returns:
        torch.tensor: Noisy image.
    """
    mapped_img = img + 1.0
    noise = torch.poisson(mapped_img * lamb)
    noisy_img = mapped_img + noise
    noisy_img = noisy_img - 1.0
    return torch.clamp(noisy_img, min=-1.0, max=1.0)


def get_speckle_noisy_img(img, noise_level=0.2):
    """
    Add speckle noise to a normalized image tensor in the range [-1, 1].

    Args:
        img (torch.Tensor): Input image tensor.
        noise_level (float): Noise intensity.

    Returns:
        torch.Tensor: Noisy image tensor.
    """
    min_val = -1.0
    max_val = 1.0
    mapped_img = (img - min_val) / (max_val - min_val)
    noise = torch.randn_like(mapped_img) * noise_level
    noisy_mapped_img = mapped_img + noise * mapped_img
    noisy_img = noisy_mapped_img * (max_val - min_val) + min_val
    return torch.clamp(noisy_img, min=-1.0, max=1.0)


def get_jpeg_compression_img(img, quality=80):
    """
    Apply JPEG compression noise to a normalized image tensor.

    Args:
        img (torch.Tensor): Image tensor with shape [1, C, H, W] in [-1, 1].
        quality (int): JPEG quality (0–100).

    Returns:
        torch.Tensor: Compressed image tensor in shape [1, C, H, W].
    """
    image_np = 255.0 * (img[0].cpu().numpy() + 1.0) / 2.0
    pil_image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    compressed_image = np.array(Image.open(buffer)).astype(np.float32) / 255.0
    compressed_img = compressed_image * 2.0 - 1.0
    return torch.from_numpy(compressed_img.transpose(2, 0, 1)).unsqueeze(0).cuda()


def get_dropout_img(img, drop_p=0.2):
    """
    Apply spatial dropout to an image tensor.

    Args:
        img (torch.Tensor): Input image tensor.
        drop_p (float): Dropout probability.

    Returns:
        torch.Tensor: Image tensor after dropout.
    """
    return F.dropout(img, p=drop_p, training=True, inplace=True) * (1 - drop_p)


class ResNetImageQuality(nn.Module):
    """
    ResNet-based image quality assessment model.
    Outputs a single quality score in [0, 1] using sigmoid activation.
    """
    def __init__(self, pretrained=True):
        super(ResNetImageQuality, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def unpad_to_original(x_padded: torch.Tensor, padding_size: tuple) -> torch.Tensor:
    """
    Remove padding from a tensor to restore the original dimensions.

    Args:
        x_padded (torch.Tensor): Padded tensor of shape [B, C, H, W].
        padding_size (tuple): Tuple of (pad_height, pad_width) to remove.

    Returns:
        torch.Tensor: Unpadded tensor.
    """
    pad_h, pad_w = padding_size
    return x_padded[:, :, :x_padded.size(2) - pad_h, :x_padded.size(3) - pad_w]


def pad_to_multiple_of_32(x_orig: torch.Tensor) -> tuple:
    """
    Pad a tensor so that height and width are multiples of 32.

    Args:
        x_orig (torch.Tensor): Input tensor of shape [B, C, H, W].

    Returns:
        tuple: (padded_tensor, (pad_height, pad_width))
    """
    _, _, h, w = x_orig.size()
    pad_h = (32 - h % 32) if h % 32 != 0 else 0
    pad_w = (32 - w % 32) if w % 32 != 0 else 0
    x_padded = F.pad(x_orig, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)
    return x_padded, (pad_h, pad_w)


def random_remove_patch(x_orig: torch.Tensor, window_size: tuple) -> tuple:
    """
    Randomly remove a rectangular patch (set to zero) from the input tensor.

    Args:
        x_orig (torch.Tensor): Input tensor of shape [B, C, H, W].
        window_size (tuple): Size (height, width) of the patch to remove.

    Returns:
        tuple:
            - torch.Tensor: Tensor with patch set to zero.
            - tuple: (top, left, patch_height, patch_width) patch location and size.
    """
    _, _, H, W = x_orig.shape
    patch_h, patch_w = window_size

    if patch_h > H or patch_w > W:
        raise ValueError("window_size exceeds image dimensions")

    top = torch.randint(0, H - patch_h + 1, (1,)).item()
    left = torch.randint(0, W - patch_w + 1, (1,)).item()

    x_modified = x_orig.clone()
    x_modified[:, :, top:top + patch_h, left:left + patch_w] = 0.0

    return x_modified, (top, left, patch_h, patch_w)


def down_up_sample(x_orig: torch.Tensor, sample_rate: float):
    _, _, H, W = x_orig.shape
    new_H = max(1, int(H//sample_rate))
    new_W = max(1, int(W//sample_rate))
    
    x_down = F.interpolate(x_orig, size=(new_H, new_W), mode='bilinear', align_corners=False)
    x_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)
    return x_up



class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified):
        cls_fn = None

        if self.config.model.type == 'openai':
            model = create_model_lung_ct_v1()
            # if self.config.model.use_fp16:
            #     model.convert_to_fp16()

            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        if simplified:
            print('Run Simplified DDNM, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            avg_psnr, psnr_list = self.simplified_ddnm_plus(model, cls_fn)
            return avg_psnr, psnr_list
            
    def simplified_ddnm_plus(self, model, cls_fn):
        import torch
        print('##### simplified_ddnm_plus #####')
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)
        # print('test_dataset ---===> ',len(test_dataset))
        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        # print("args.deg:",args.deg)
        if args.deg =='denoising':
            A = lambda z: z
            Ap = A
        elif args.deg =='diy':
            A = lambda z: z
            Ap = A
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        psnr_list = []
        
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes, file_name in pbar:
            x_orig = x_orig.to(self.device)
            
            x_orig = data_transform(self.config, x_orig)
            x_orig, padding_size = pad_to_multiple_of_32(x_orig)
            # print('x_orig ---> ',x_orig.shape)

            # x_orig, patch_info = random_remove_patch(x_orig, window_size=(64,64))
            x_orig = down_up_sample(x_orig, sample_rate=4)
            # print('x_orig ---> ',x_orig.shape)
            y = A(x_orig)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            if self.args.add_noise: # for denoising test
                # print('self.args.noise_rate ---> ',self.args.noise_rate)
                y = get_gaussian_noisy_img(y, self.args.noise_rate)

            Apy = Ap(y)

            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                # print('N :::::: ',n)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                # print('time_pairs :::::: ',time_pairs)

                from transformers import BertTokenizer, BertModel
                import torch
                def get_ddpm_embedding():
                    device = torch.device("cuda")
                    bert_path = "chinesebert"
                    text = [
                    # 'window width and window level is 1500 and -500',
                    'window width and window level is 400 and 0'
                    ]
                    tokenizer = BertTokenizer.from_pretrained(bert_path)
                    bert_model = BertModel.from_pretrained(bert_path)
                    bert_model.to(device)
                    
                    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                    
                    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
                    
                    embedding = cls_embedding.unsqueeze(1)
                    
                    return embedding

                prompt_info_used = get_ddpm_embedding()


                lambda_ts = []
                sigma_ts = []
                at_nexts = []
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip

                    if i<=10000000: #self.args.lambda_t:

                        if j<0: j=-1 

                        if j < i: # normal sampling 
                            t = (torch.ones(n) * i).to(x.device)
                            next_t = (torch.ones(n) * j).to(x.device)
                            at = compute_alpha(self.betas, t.long())
                            at_next = compute_alpha(self.betas, next_t.long())
                            sigma_t = (1 - at_next**2).sqrt()
                            xt = xs[-1].to('cuda')

                            # Eq. 19

                            if t>self.args.lambda_t:
                                lambda_t = torch.ones(1).cuda()
                                x0_t_hat =  y
                                x0_t = y
                            else:
                                # lambda_t = 1-(t-self.args.lambda_t)/(1000-self.args.lambda_t)
                                lambda_t = (t)/(self.args.lambda_t)
                                lambda_t = lambda_t*torch.ones(1).cuda()

                                # print('text_embeding, attention_mask ---> ',text_embeding.shape, attention_mask.shape)
                                # t, prompt_info_used --->  torch.Size([1, 3, 288, 256]) torch.Size([1, 1, 768])    
                                # print('xt, prompt_info_used ---> ',xt.shape, prompt_info_used.shape)
                                et_tuple = model(xt, t, prompt_info_used)
                                et = et_tuple[0]
                                # print('et ---> ',et)
                                if et.size(1) == 6:
                                    et = et[:, :3]
                                # Eq. 12
                                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                                # Eq. 17
                                x0_t_hat = x0_t - lambda_t*(x0_t - y)

                            lambda_ts.append(lambda_t.cpu().numpy() if not isinstance(lambda_t, float) else lambda_t)
                            sigma_ts.append(sigma_t.cpu().numpy()[0, 0, 0, 0])
                            at_nexts.append(at_next.cpu().numpy()[0, 0, 0, 0])

                            eta = self.args.eta

                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                            # different from the paper, we use DDIM here instead of DDPM
                            xt_next = at_next.sqrt() * x0_t_hat # + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                            x0_preds.append(x0_t.to('cpu'))
                            xs.append(xt_next.to('cpu'))    
                        else: # time-travel back
                            pass


                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            # print(self.args.image_folder)
            
            x_save = x[0]
            x_save = unpad_to_original(x_save, padding_size)
            # tvu.save_image(   x[0], SAVE_PATH )

            gt_path = self.args.input_folder
            # orig = inverse_data_transform(config, x_orig[0])
            import tifffile as tiff
            gt_img = tiff.imread(gt_path+'//'+file_name[0])
            img_save = x_save.cpu().detach().numpy()
            if self.args.lambda_t==0:
                img_save = np.squeeze(x_orig.cpu().detach().numpy())
            img_save = np.mean(img_save, axis=0)
            img_save_raw = img_save
            # img_save = img_save/np.mean(img_save)*np.mean(gt_img)

            gt_max = np.max(gt_img)
            gt_min = np.min(gt_img)

            img_save_max = np.max(img_save)
            img_save_min = np.min(img_save)

            gt_img = (gt_img-gt_min)/(gt_max-gt_min)
            img_save = (img_save-img_save_min)/(img_save_max-img_save_min)

            # mask_img = get_brain_mask(gt_img)   
            mask_img = 1         
            gt_img = gt_img*mask_img
            img_save = img_save*mask_img
            # img_save = img_save/np.mean(img_save)*np.mean(gt_img)
            mse = np.mean((img_save - gt_img) ** 2)
            gt_energy = np.mean((gt_img) ** 2)
            if 0:
                psnr = 10 * np.log10(gt_energy / mse)
            psnr = 10 * np.log10(255*255 / mse)

            SAVE_PATH = os.path.join(self.args.image_folder, file_name[0])
            SAVE_PATH = SAVE_PATH+'_'+str(self.args.lambda_t)+'_'+str(round(psnr,2))+'.tif'
            print('SAVE PATH ---> ',SAVE_PATH)
            import tifffile
            tifffile.imwrite(SAVE_PATH, img_save_raw)

            if 0:
                mask_folder = self.args.output_path+'//'+self.args.output_folder+'//0_mask'
                os.makedirs(mask_folder, exist_ok=True)
                SAVE_MASK = mask_folder+'//'+file_name[0]+'_mask.tif'
                tifffile.imwrite(SAVE_MASK, mask_img)

            avg_psnr += psnr
            psnr_list.append(psnr)
            idx_so_far += y.shape[0]
            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        plt.plot(range(len(lambda_ts)), lambda_ts, label='lambda_t', marker='o', linestyle='-')
        plt.plot(range(len(sigma_ts)), sigma_ts, label='sigma_t', marker='x', linestyle='--')
        plt.plot(range(len(at_nexts)), at_nexts, label='at_next', marker='s', linestyle='-.')
        plt.legend()
        # plt.show()
        if not os.path.exists('lambda_t'): 
            os.mkdir('lambda_t')
        plt.savefig(f"lambda_t/ts_{sigma_y / 2}_{self.args.lambda_t}.png")

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        return avg_psnr, psnr_list
    
 

import cv2
def get_brain_mask(image1, index=0, if_save=0):
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io, measure, morphology, filters
    # gray = image1
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    import copy
    image = copy.deepcopy(image1)
    thresh = filters.threshold_otsu(image) #*0.2
    top_left_10x10 = image[0:10, 0:10]
    bg_value = top_left_10x10.mean()
    linear_rate = 0.2
    thresh = thresh*linear_rate+bg_value*(1-linear_rate)

    binary = image > thresh
    binary = np.uint8(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(binary, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    filled_max_region = cv2.drawContours(binary, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    if_save = 0
    if if_save:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(filled_max_region, cmap='gray')
        ax[1].set_title('Mask of the Largest Connected Component')
        ax[1].axis('off')

        mask_folder = '0_mask'
        os.makedirs(mask_folder, exist_ok=True)
        from datetime import datetime
        current_datetime = datetime.now()
        datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        mask_name = mask_folder+'//'+str(index)+'.png'
        plt.savefig(mask_name, bbox_inches='tight')
        plt.show()
    return filled_max_region



# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    # print('t : ',t,' beta : ',beta,' a : ',a)
    return a
