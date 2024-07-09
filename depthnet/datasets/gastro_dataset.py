import torch
import random
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os

from PIL import Image, ImageFile
import skimage.transform
import numpy as np
import PIL.Image as pil
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import collections

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
def pil_depth_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('I')
        
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, collections.abc.Mapping):  # Some custom condition
        processed_batch = {}
        for key in elem:
            if "correspondences" in key:
                processed_batch[key] = [d[key] for d in batch]
            else:
                processed_batch[key] = default_collate([d[key] for d in batch])
        try:
            return elem_type(processed_batch)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return processed_batch
    else:  # Fall back to `default_collate`
        return default_collate(batch)
    
def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    num_epochs = cfgs.get('num_epochs', 30)
    num_workers = cfgs.get('num_workers', 8)
    data_path = cfgs.get('data_path')
    height = cfgs.get('height', 256)
    width = cfgs.get('width', 320)
    frame_ids = cfgs.get('frame_ids', [0,-1,1])
    num_scales = len(cfgs.get('scales', [0,1,2,3]))
    dataset = globals().get(cfgs.get('dataset', C3VDDataset))
    split = cfgs.get('split')
    fpath = os.path.join("splits", split, "{}_files.txt")
    img_ext = '.png' if cfgs.get('png', False) else '.jpg'
    load_depth = cfgs.get('load_depth', False)
    
    train_filenames = readlines(fpath.format("train"))
    matcher_result_load_train = cfgs.get('matcher_result_train', None)
    if matcher_result_load_train:
        matcher_result_load_train = np.load(matcher_result_load_train, allow_pickle=True).all()
    train_dataset = dataset(data_path, train_filenames, matcher_result_load_train,
        height, width, frame_ids, num_scales,
        is_train=True, img_ext=img_ext, load_depth=load_depth)
    train_loader = DataLoader(
        train_dataset, batch_size, True, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    info_dict = {"num_total_steps": len(train_filenames) // batch_size * num_epochs}
    
    loaders_dict = {"train_loader": train_loader, "info_dict": info_dict}
    
    val_dataset, test_dataset = ([], [])
    
    if cfgs.get('run_val', False):
        val_filenames = readlines(fpath.format("val"))
        matcher_result_load_val = cfgs.get('matcher_result_val', None)
        if matcher_result_load_val:
            matcher_result_load_val = np.load(matcher_result_load_val, allow_pickle=True).all()
        val_dataset = dataset(data_path, val_filenames, matcher_result_load_val,
            height, width, frame_ids, num_scales,
            is_train=False, img_ext=img_ext, load_depth=load_depth)
        val_loader = DataLoader(
            val_dataset, batch_size, True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=False, drop_last=True)
        loaders_dict["val_loader"] = val_loader
        
    if cfgs.get('run_test', False):
        test_filenames = readlines(fpath.format("test"))
        matcher_result_load_test = cfgs.get('matcher_result_test', None)
        if matcher_result_load_test:
            matcher_result_load_test = np.load(matcher_result_load_test, allow_pickle=True).all()
        test_dataset = dataset(data_path, test_filenames, matcher_result_load_test,
            height, width, frame_ids, num_scales,
            is_train=False, img_ext=img_ext, load_depth=load_depth)
        test_loader = DataLoader(
            test_dataset, batch_size, True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=False, drop_last=True)
        loaders_dict["test_loader"] = test_loader
    
    print("Using split:\n  ", split)
    print("There are {:d} training items, {:d} validation items and {:d} tesing items\n".format(
        len(train_dataset), len(val_dataset), len(test_dataset)))
    
    return loaders_dict

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
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False, 
                 load_pose=False):
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
        self.load_depth = load_depth
        self.load_pose = load_pose

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

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
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
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

        inputs.update(self.load_extra(do_flip, index))

        for i in self.frame_idxs:
            inputs[('color', i, -1)] = self.get_color(folder, line[2+i], do_flip)    
            if self.load_depth and i == 0: # only frame 0 need depth_gt
                depth = self.get_depth(folder, line[2+i], do_flip).resize((self.width, self.height))
                depth = self.postprocess_depth(depth)
                inputs[('depth_gt')] = depth
                
            if self.load_pose:
                inputs[('pose_gt', i, 0)] = self.get_pose(folder, line[2+i], do_flip)

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
            del inputs[("color_aug", i, -1)]

        return inputs
        
    def get_color(self, folder, frame_index_str, do_flip):
        raise NotImplementedError
    
    def get_depth(self, folder, frame_index_str, do_flip):
        raise NotImplementedError
    
    def get_pose(self, folder, frame_index_str, do_flip):
        raise NotImplementedError
    
    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth))
    
    def load_extra(self, do_flip, index):
        return {}
    
class C3VDDataset(MonoDataset):
    def __init__(self,
                 data_path,
                 filenames,
                 correspondences,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False, 
                 load_pose=False):
        super(C3VDDataset, self).__init__(data_path, filenames, height, width, frame_idxs,
                 num_scales, is_train=is_train, img_ext=img_ext, load_depth=load_depth, load_pose=load_pose)
        self.K = np.array([[0.56959306, 0, 0.5, 0],
                           [0, 0.71185083, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (1350, 1080) # reshape the whole dataset before hand. raw shape is [w=1350, h=1080]
        self.correspondences = correspondences #kwargs['correspondences']
        self.loader_depth = pil_depth_loader
    
    def get_color(self, folder, frame_index_str, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index_str))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index_str):
        f_str = "{}_color{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def get_depth_path(self, folder, frame_index_str):
        f_str = "{}_depth{}".format(frame_index_str, ".tiff")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def get_depth(self, folder, frame_index_str, do_flip):
        depth = self.loader_depth(self.get_depth_path(folder, frame_index_str))
        
        if do_flip:
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)
        # depth = np.array(depth)/(2**16-1)
        return depth
    
    def get_pose(self, folder, frame_index_str, do_flip):
        pose_path = os.path.join(self.data_path, folder, "pose.txt")
        with open(pose_path, 'r') as f:
            lines = f.read().splitlines()
        pose = lines[int(frame_index_str)].split(",")
        pose = np.array(pose, dtype=float)
        pose = pose.reshape(4, 4)
        return pose
    
    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth) / (2**16 - 1))[None, ...]
    
    def load_extra(self, do_flip, index):
        inputs = {}
        if self.correspondences:
            if do_flip:
                flip_str = 'do_flip'
            else:
                flip_str = 'no_flip'
        
            for idx, fid in [[0, -1], [1, 1]]:
                source_fid = 0
                inputs[("correspondences", source_fid, fid)] = self.correspondences[flip_str][index][idx]
        return inputs


class SimcolDataset(C3VDDataset):
    def __init__(self, *args, **kwargs):
        super(SimcolDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (475, 475) # reshape the whole dataset before hand
        
    
    def get_image_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def get_depth_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str.replace("FrameBuffer", "Depth"), ".png")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth) / (255 * 256))[None, ...]
    

class NYUDataset(C3VDDataset):
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        # 518.8579 0 325.58245 - 41
        # 0 519.46961 253.73617 - 45
        # 0 0 1
        
        # 518.8579 0 284.58245 
        # 0 519.46961 208.73617
        # 0 0 1
        
        self.K = np.array([[518.8579 / 560, 0, 284.58245 / 560, 0],
                           [0, 519.46961 / 426, 208.73617 / 426, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (560, 426) # reshape the whole dataset before hand
        
    
    def get_image_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def get_depth_path(self, folder, frame_index_str):
        f_str = "{}{}".format("depth/" + frame_index_str, ".png")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
    
    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth) / 5000)[None, ...]
    
class TestNYUDataset(NYUDataset):
    def __init__(self,                 
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs=[-1],
                 num_scales=1,
                 is_train=False,
                 img_ext='.png',
                 load_depth=False, 
                 load_pose=False):
                    
        super(NYUDataset, self).__init__(data_path, filenames, None, height, width, frame_idxs,
                 num_scales, is_train=is_train, img_ext=img_ext, load_depth=load_depth, load_pose=load_pose)
            