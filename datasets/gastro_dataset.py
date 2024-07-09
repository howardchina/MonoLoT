import torch
import random
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os

from PIL import Image
import skimage.transform
import numpy as np
import PIL.Image as pil

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
class MonoDataset(torch.utils.data.Dataset):
    """MonoDataset
    
    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self, data_path, filenames, height, width, frame_idxs,
                 num_scales, is_train=False, img_ext='.jpg'):
        super(MonoDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.ANTIALIAS
        self.interp = torchvision.transforms.InterpolationMode.LANCZOS
        
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        
        self.loader = pil_loader
        self.to_tensor = torchvision.transforms.ToTensor()
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = torchvision.transforms.Resize((self.height // s, self.width // s),
                                                           interpolation=self.interp)
        
    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        """Return a single training item from the dataset as a dictionary
        
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
        
            ("color", <frame_id>, <scale>)      for raw colour images,
            ("K", scale) or ("inv_K", scale)    for camera intrinsics,
            "stereo_T"                          for camera extrinscis
            
        <frame_id> is an integer (e.g. 0, -1, or 1) representing the emporal step relative to 'index'
        
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        line = self.filenames[index].split()
        folder = line[0]
        
        for i in self.frame_idxs:
            inputs[('color', i, -1)] = self.get_color(folder, line[2+i], do_flip)
        
        for scale in range(self.num_scales):
            K = self.K.copy()
            
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            
            inv_K = np.linalg.pinv(K)
            
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        if do_color_aug:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            def color_aug(img):
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F.adjust_hue(img, hue_factor)
                return img
        else:
            color_aug = (lambda x: x)
        
        self.preprocess(inputs, color_aug)
        
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
        
        return inputs
        
    def get_color(self, folder, frame_index_str, do_flip):
        raise NotImplementedError
    

class GastroDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(GastroDataset, self).__init__(*args, **kwargs)
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # array([[630.74067784,   0.        , 515.66356656],
        #  [  0.        , 630.19314286, 446.75170578],
        #  [  0.        ,   0.        ,   1.        ]])
        self.K = np.array([[0.616, 0, 0.5, 0],
                           [0, 0.708, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = () # reshape the whole dataset before hand
        
    def get_color(self, folder, frame_index_str, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index_str))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_image_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

class SimcolDataset(GastroDataset):
    def __init__(self, *args, **kwargs):
        super(SimcolDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (475, 475) # reshape the whole dataset before hand


class C3VDDataset(GastroDataset):
    def __init__(self, *args, **kwargs):
        super(C3VDDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.56959306, 0, 0.5, 0],
                           [0, 0.71185083, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (1350, 1080) # reshape the whole dataset before hand. raw shape is [w=1350, h=1080]
    
    def get_image_path(self, folder, frame_index_str):
        f_str = "{}_color{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path