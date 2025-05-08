import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
# from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
from guided_diffusion.script_util import create_model_MRI
import random

from scipy.linalg import orth
from skimage.util import random_noise
from PIL import Image
import io
import torch.nn.functional as F
import tifffile
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


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
    # 1. 将原始张量映射到 [0, 1] 范围
    min_val = -1.0
    max_val = 1.0
    mapped_img = (img - min_val) / (max_val - min_val)

    # 2. 生成 [0, 1] 范围内的Speckle噪声
    # Speckle噪声通常服从均值为0的指数分布
    noise = torch.randn_like(mapped_img) * noise_level

    # 3. 将噪声添加到映射后的图像上
    noisy_mapped_img = mapped_img + noise * mapped_img

    # 4. 将添加噪声后的图像映射回 [-1, 1] 范围
    noisy_img = (noisy_mapped_img * (max_val - min_val)) + min_val
    return torch.clamp(noisy_img, min=-1.0, max=1.0)


def get_jpeg_compression_img(img, quality=80):
    """
    Add JPEG compression noise to the image.

    Args:
        image (numpy.ndarray): Input image.
        quality (int): JPEG compression quality level (0-100).

    Returns:
        numpy.ndarray: Noisy image.
    """
    image = 255.0 * (img[0].cpu().numpy() + 1.0) / 2.0
    # Convert the NumPy array to PIL Image
    # if len(image.shape) > 2:
    #     return None
    pil_image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))

    # Create an in-memory buffer to store the compressed image
    buffer = io.BytesIO()

    # Save the PIL Image to the buffer with JPEG compression
    pil_image.save(buffer, format='JPEG', quality=quality)

    # Load the compressed image from the buffer
    compressed_image = np.array(Image.open(buffer))

    # Convert the image back to the range of [0, 1]
    compressed_image = compressed_image.astype(np.float32) / 255.0
    compressed_img = compressed_image * 2.0 - 1.0
    return torch.from_numpy(compressed_img.transpose(2, 0, 1)).unsqueeze(0).cuda()


def get_dropout_img(img, drop_p=0.2):
    return F.dropout(img, p=drop_p, training=True, inplace=True) * (1 - drop_p)


class ResNetImageQuality(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetImageQuality, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(pretrained=pretrained)
        # 修改最后一层全连接层，将输出维度更改为1
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        # 使用Softmax
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.softmax(x)
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
        # print('config.diffusion.beta_schedule : ',config.diffusion.beta_schedule)
        # print('config.diffusion.beta_start : ',config.diffusion.beta_start)
        # print('config.diffusion.beta_end : ',config.diffusion.beta_end)
        # print('config.diffusion.num_diffusion_timesteps : ',config.diffusion.num_diffusion_timesteps)
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        # print('betas : ',betas[0::50])
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
            config_dict = vars(self.config.model)
            # model = create_model(**config_dict)
            model = create_model_MRI()
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

        # args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        psnr_list = []
        '''
        val_loader 是一个迭代器，通常在机器学习和数据处理中，它用于在验证集上迭代数据，val_loader 可能是一个加载验证数据集的 DataLoader 对象。
        tqdm.tqdm() 是 tqdm 模块中的主要函数，它接受一个可迭代对象作为参数，并返回一个迭代器，该迭代器会在每次迭代时更新进度条。
        pbar 是 tqdm.tqdm(val_loader) 返回的进度条对象的引用。你可以在循环中使用 pbar 来追踪进度
        '''
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes, file_name in pbar:
            x_orig = x_orig.to(self.device)
            # print('classes ---> ',classes, file_name)
            # print('x_orig ---> ',x_orig.shape)
            # 只是做一些归一化无关tensor形状
            x_orig = data_transform(self.config, x_orig)
            # print('x_orig ---> ',x_orig.shape)

            y = A(x_orig)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            if self.args.add_noise: # for denoising test
                # print('self.args.noise_rate ---> ',self.args.noise_rate)
                y = get_gaussian_noisy_img(y, self.args.noise_rate)

            Apy = Ap(y)

            # self.args.lambda_t
            # print('len(Apy) :::::: ',len(Apy))
                
            # init x_T
            '''
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            '''
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                y.shape[2],
                y.shape[3],
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

                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("guided_diffusion//bert-base//models--bert-base-cased//snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e")
                def get_mod_prompt(tokenizer, mod_cls=0):
                    raw_inputs = [
                            "The format of input data is t1 mri.",
                        ]
                    res = tokenizer(raw_inputs, padding="max_length", max_length=16, return_tensors="pt")
                    text_embeding = res["input_ids"][mod_cls].view(1,-1,1)
                    token_type_ids = res["token_type_ids"][mod_cls].view(1,-1,1)
                    attention_mask = res["attention_mask"][mod_cls].view(1,-1,1)
                    # "The format of input data is ct.",
                    # "The format of input data is t1 mri.",
                    #  "The format of input data is t2 mri."
                    return text_embeding, attention_mask
                
                text_embeding, attention_mask = get_mod_prompt(tokenizer)

                A = lambda z: z
                Ap = A

                lambda_ts = []
                sigma_ts = []
                at_nexts = []
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip

                    if j<0: j=-1 

                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next**2).sqrt()
                        xt = xs[-1].to('cuda')
                        # print('\n xt ---> ',xt.shape)

                        # et = model(xt, t)
                        et_tuple = model(xt, t, text_embeding, attention_mask)
                        et = et_tuple[0]
                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        # Eq. 19
                        # print('\n',sigma_t[0,0,0,0].cpu().detach().numpy(), \
                        # at_next[0,0,0].cpu().detach().numpy(), sigma_y)
                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.

                        # Eq. 17
                        # print('\n x0_t ---> ',x0_t.shape, ' y ---> ',y.shape)
                        x0_t_hat = x0_t - lambda_t*(x0_t - y)

                        eta = self.args.eta

                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                        # different from the paper, we use DDIM here instead of DDPM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))    
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            # print(self.args.image_folder)
            

            # tvu.save_image(   x[0], SAVE_PATH )

            gt_path = '..//0_united_image//GT_tif'
            # orig = inverse_data_transform(config, x_orig[0])
            import tifffile as tiff
            gt_img = tiff.imread(gt_path+'//'+file_name[0])
            img_save = x[0].cpu().detach().numpy()
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

            mask_img = get_brain_mask(gt_img)            
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
            # print('SAVE PATH ---> ',SAVE_PATH, x[0].shape, torch.max(x[0]))
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

        # 显示原始图像
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # 显示mask
        ax[1].imshow(filled_max_region, cmap='gray')
        ax[1].set_title('Mask of the Largest Connected Component')
        ax[1].axis('off')

        # 保存绘制的图像
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
    # cumpord 返回矩阵元素累计乘积
    # index_select 从张量的某个维度抽取
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    # print('t : ',t,' beta : ',beta,' a : ',a)
    return a
