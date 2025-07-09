import sys
import torch
import safetensors
from torch import nn
from typing import List
from transformers import BertTokenizer, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        # print(type(features))
        # print(type(features[0]))
        nfeatures = []
        for f in features:
            if f is not None:
                if type(f) in [list, tuple]:
                    for ff in f:
                        # print(1, type(ff))
                        # print(2, ff.shape)
                        pass
                        # nfeatures.append(ff.detach().float())
                else:
                    nfeatures.append(f.detach().float())
            else:
                nfeatures.append(None)
        setattr(module, name, nfeatures)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass

class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations
        # for idx, block in enumerate(self.model.up_blocks): #  down_blocks
        for idx, block in enumerate(self.model.down_blocks):
            if idx in blocks[0]:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

        for idx, block in enumerate(self.model.up_blocks):
            if idx in blocks[1]:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)
            # print(idx)
        
        bert_path = "config.bert_path"
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertModel.from_pretrained(bert_path)
        texts = [
            # 'window width and window level is 1500 and -500.',
            'window width and window level is 400 and 0.'
        ]
        cancer_texts = [texts[0]]
        
        for i, text in enumerate(cancer_texts):
            prompt_embeding = []
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            prompt_embeding.append(cls_embedding.view(1, 1, -1))

            prompt_embeding = prompt_embeding * 4 # config.batch_size
            prompt_embeding = torch.cat(prompt_embeding, dim=0)
        self.prompt_embeding = prompt_embeding

    def _load_pretrained_model(self, model_path, **kwargs):
        from diffusers import UNet2DModel, UNet2DConditionModel
        from diffusers import DDPMScheduler
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.model = UNet2DConditionModel(
            sample_size=256,  
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

        pretrained_w = safetensors.torch.load_file(model_path + "/diffusion_pytorch_model.safetensors")
        self.model.load_state_dict(pretrained_w)
        print(f"\033[91mPretrained model is successfully loaded from {model_path}\033[0m")

        self.model.conv_out = nn.Conv2d(64, 2, kernel_size=3, padding=1) 

    # @torch.no_grad()
    def forward(self, xs, noise=None):
        activations = []
        input_with_noise = []
        for t in self.steps:
            for x in xs:
                t = torch.tensor([t]).to(x.device).long()
                noisy_x = self.noise_scheduler.add_noise(x, noise, t)
                input_with_noise.append(noisy_x)
                direct_out = self.model(noisy_x, t, encoder_hidden_states=self.prompt_embeding.to(xs.device), return_dict=False)[0]

                for block in self.feature_blocks:
                    if type(block.activations) is list:
                        activations.append(*block.activations)
                    else:
                        activations.append(block.activations)
                    block.activations = None

        return activations, input_with_noise, direct_out

def collect_features(config, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple((config.image_size, config.image_size))
    resized_activations = []
    for feats in activations:
        feats = nn.functional.interpolate(
            feats, size=size, mode=config.upsample_mode
        )
        resized_activations.append(feats)
    # import pdb
    # pdb.set_trace()
    return torch.cat(resized_activations, dim=1)

def prepare_fea(fea, label):
    d = fea.shape[1]
    fea = fea.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    label = label.flatten()
    return fea, label
