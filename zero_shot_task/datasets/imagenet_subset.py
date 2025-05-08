import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    # print('pil_loader PATH ---> ',path)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    

import tifffile as tiff
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    elif '.tif' in path:
        img = tiff.imread(path)
        return img
    else:
        return pil_loader(path)



import os
def list_files_in_directory(directory_path):
    file_list = []
    entries = os.listdir(directory_path)
    for entry in entries:
        full_path = os.path.join(directory_path, entry)  
        if os.path.isfile(full_path):  
            file_list.append(entry)

    return file_list



class ImageDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 image_size=128,
                 normalize=True,
                 test_data_num=1,
                 tag_name = ''):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                '''
                self.transform = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
                '''
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s" % meta_file)
        self.num = test_data_num
        self.metas = []
        self.classifier = None
        # suffix = ".jpeg"
        suffix = ""
        # print('root_dir ---> ',root_dir)
        img_name_list = list_files_in_directory(root_dir)
        print(tag_name, ' ---> ',img_name_list)
        def filter_list_by_inclusion(original_list, string_to_include):
            return [item for item in original_list if string_to_include in item]
        # img_name_list = filter_list_by_inclusion(img_name_list, 'registered')
        '''
        show_index_list = range(140, 280, 20) #140 180 220 260
        img_name_list_figure = []
        for show_index in show_index_list:
            img_name_list_sub = filter_list_by_inclusion(img_name_list, str(show_index))
            img_name_list_figure = img_name_list_figure+img_name_list_sub

        img_name_list_figure = filter_list_by_inclusion(img_name_list_figure, 'sub-01_ses-1_T1w_defaced_registered_layer_240')
        '''
        # img_name_list_figure = filter_list_by_inclusion(img_name_list, '2022101101_T1_layer_10') T1 2022101101_FLAIR_layer_10
        img_name_list_figure = filter_list_by_inclusion(img_name_list, tag_name)


        # img_name_list_figure = img_name_list
        if test_data_num<len(img_name_list_figure):
            img_name_list_figure = img_name_list_figure[0:test_data_num]
        self.num = len(img_name_list_figure)
        print('img_name_list_figure ---> ',img_name_list_figure, len(img_name_list_figure))
        # assert len(img_name_list_figure)<10, "The number must be positive"
        # print('lines ---> ',lines)
        for img_name in img_name_list_figure:
            self.metas.append((img_name + suffix, -1))
        # print('metas ---> ',len(self.metas))
        # print("read meta done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        # print('filename ---> ',filename)
        img = default_loader(filename)
        thres = 1200
        img[img > thres] = thres
        import numpy as np
        if np.max(img)>1:
            img = img/np.max(img)
        # img --->  (256, 256)
        # print('img ---> ',img.size)

        # transform
        if self.transform is not None:
            # print('GET TRANSFORM ~~~~~', self.transform)
            img = self.transform(img)
        
        # print('img ---> ',img.shape)
        if img.shape[0]==1:
            img = img.repeat(3, 1, 1)
        # print('img ---> ',img.shape)
        img = img.float()
        import torch
        # print(img.dtype, ' MAX ==== ',torch.max(img))
        return img, cls, self.metas[idx][0]