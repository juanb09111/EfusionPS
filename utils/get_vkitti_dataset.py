# %%
import scipy.io
import os.path
import math
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config_kitti
from utils.lidar_cam_projection import read_calib_file, load_velo_scan, full_project_velo_to_cam2, project_to_image
import glob
import cv2
import matplotlib.pyplot as plt
import random
# %%



def get_vkitti_files(dirName, ext):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)

    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_vkitti_files(fullPath, ext)
        elif fullPath.find("morning") != -1 and fullPath.find("Camera_0") != -1 and fullPath.find(ext) != -1:
            allFiles.append(fullPath)

    return allFiles



class vkittiDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_root, depth_root, transforms, n_samples=None):

        self.imgs_root = imgs_root
        self.depth_root = depth_root

        images = get_vkitti_files(imgs_root, "jpg")
        depth_files = get_vkitti_files(depth_root, "png")


        self.transforms = transforms
        if n_samples is None:
            self.source_imgs = images
            self.depth_imgs = depth_files
        else:
            self.source_imgs = images[:n_samples]
            self.depth_imgs = depth_files[:n_samples]

        print("Training/evaluating on {} samples".format(len(self.source_imgs)))

    def find_k_nearest(self, lidar_fov):
        k_number = config_kitti.K_NUMBER
        b_lidar_fov = torch.unsqueeze(lidar_fov, dim=0)

        distances = torch.cdist(b_lidar_fov, b_lidar_fov, p=2)
        _, indices = torch.topk(distances, k_number + 1, dim=2, largest=False)
        indices = indices[:, :, 1:]  # B x N x 3

        return indices.squeeze_(0).long()
    
    def sample_depth_img(self, depth_tensor):
        (img_height, img_width) = depth_tensor.shape[1:]
        
        rand_x_coors = []
        rand_y_coors = []

        for i in range(0, config_kitti.N_NUMBER*2):
            rand_x_coors.append(random.randint(0, img_width -1))

        for k in range(0, config_kitti.N_NUMBER*2):
            rand_y_coors.append(random.randint(0, img_height -1))

        coors = torch.zeros((config_kitti.N_NUMBER*2, 2))

        # coors in the form of NxHxW
        coors[:, 1] = torch.tensor(rand_x_coors, dtype=torch.long)
        coors[:, 0] = torch.tensor(rand_y_coors, dtype=torch.long)
        coors = torch.tensor(coors, dtype=torch.long)

       
        # find unique coordinates
        _, indices = torch.unique(coors[:, :2], dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1

        imPts = coors[unique_indices]

        depth = depth_tensor[0, imPts[:,0], imPts[:,1]]/256
        
        #filter out long ranges of depth
        inds= depth<config_kitti.MAX_DEPTH

        # fig = plt.figure(4)
        # # ax = plt.axes(projection="3d")
        # ax = plt.axes(projection='3d')

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # x_data = imPts[inds, 1]
        # y_data = imPts[inds, 0]
        # z_data = depth[inds]
        # ax.scatter3D(x_data, y_data, z_data, cmap='Greens', s=1)
        # plt.show()
        # imPts in NxHxW
        return imPts[inds, :][:config_kitti.N_NUMBER], depth[inds][:config_kitti.N_NUMBER]

    def __getitem__(self, index):

        img_filename = self.source_imgs[index]  # rgb image
        scene = img_filename.split("/")[-6]
        
        basname = img_filename.split(".")[-2].split("_")[-1]
        
        depth_filename = [s for s in self.depth_imgs if (scene in s and basname in s)][0]
        # print(img_filename, depth_filename)

        source_img = Image.open(img_filename)
        depth_img = Image.open(depth_filename)
        # img width and height
        

        if self.transforms is not None:
            source_img = self.transforms(crop=True)(source_img)
            depth_img = self.transforms(crop=True)(depth_img)

        # plt.imshow(source_img.permute(1,2,0))
        # plt.show()

        # plt.imshow(depth_img.permute(1,2,0))
        # plt.show()

        imPts, depth = self.sample_depth_img(depth_img)

        virtual_lidar = torch.zeros(imPts.shape[0], 3)
        virtual_lidar[:, 0:2] = imPts
        virtual_lidar[:, 2] = depth


        fig = plt.figure(4)
        # ax = plt.axes(projection="3d")
        # ax = plt.axes(projection='3d')

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # x_data = virtual_lidar[:, 1]
        # y_data = virtual_lidar[:, 0]
        # z_data = virtual_lidar[:, 2]
        # ax.scatter3D(x_data, y_data, z_data, cmap='Greens', s=1)
        # plt.show()

        mask = torch.zeros(source_img.shape[1:], dtype=torch.bool)
        mask[imPts[:, 0], imPts[:, 1]] = True
        # plt.imshow(mask)
        # plt.show()
        k_nn_indices = self.find_k_nearest(virtual_lidar)

        sparse_depth = torch.zeros_like(
            source_img[0, :, :].unsqueeze_(0), dtype=torch.float)

        sparse_depth[0, imPts[:, 0], imPts[:, 1]] = torch.tensor(
            depth, dtype=torch.float)

        print(source_img.shape, virtual_lidar.shape, mask.shape, sparse_depth.shape, k_nn_indices.shape, depth_img.shape)
        return source_img, virtual_lidar, mask, sparse_depth, k_nn_indices, depth_img, basname

    def __len__(self):
        return len(self.source_imgs)



def get_transform(resize=False, normalize=False, crop=False):
    new_size = tuple(np.ceil(x*config_kitti.RESIZE)
                     for x in config_kitti.ORIGINAL_INPUT_SIZE_HW)
    new_size = tuple(int(x) for x in new_size)
    custom_transforms = []
    if resize:
        custom_transforms.append(transforms.Resize(new_size))

    if crop:
        custom_transforms.append(
            transforms.CenterCrop(config_kitti.CROP_OUTPUT_SIZE))

    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)/255).unsqueeze(0)))
    if normalize:
        custom_transforms.append(transforms.Normalize(0.485, 0.229))
    return transforms.Compose(custom_transforms)


# imgs_root_train = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "imgs/2011_09_26/train/")
# data_depth_velodyne_root_train = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "data_depth_velodyne/train/")
# data_depth_annotated_root_train = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "data_depth_annotated/train/")

# calib_velo2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "imgs/2011_09_26/calib_velo_to_cam.txt")
# calib_cam2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "imgs/2011_09_26/calib_cam_to_cam.txt")

# kitti_dataset = kittiDataset(imgs_root_train, data_depth_velodyne_root_train,
#                              data_depth_annotated_root_train, calib_velo2cam, calib_cam2cam, transforms=get_transform, n_samples=600)
# for i in range(600):

#     # kitti_dataset.__getitem__(np.random.randint(500))
#     kitti_dataset.__getitem__(i)

def get_datasets(imgs_root, depth_root, split=False, val_size=0.20, n_samples=None):

    vkitti_dataset = vkittiDataset(imgs_root, depth_root, transforms=get_transform, n_samples=n_samples)
    if split:
        if val_size >= 1:
            raise AssertionError(
                "val_size must be a value within the range of (0,1)")

        len_val = math.ceil(len(vkitti_dataset)*val_size)
        len_train = len(vkitti_dataset) - len_val

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")
        train_set, val_set = torch.utils.data.random_split(
            vkitti_dataset, [len_train, len_val])
        return train_set, val_set
    else:
        return vkitti_dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(batch_size, imgs_root, depth_root, split=False, val_size=0.20, n_samples=None):

    if split:
        train_set, val_set = get_datasets(imgs_root, depth_root, split=True, val_size=0.20, n_samples=n_samples)

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        collate_fn=collate_fn,
                                                        drop_last=True)

        data_loader_val = torch.utils.data.DataLoader(val_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=collate_fn,
                                                      drop_last=True)
        return data_loader_train, data_loader_val

    else:
        dataset = get_datasets(imgs_root, depth_root, split=False, val_size=0.20, n_samples=n_samples)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    collate_fn=collate_fn,
                                                    drop_last=True)
    return data_loader


imgs_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_vkitti/virtual_world_vkitti-2/vkitti_2.0.3_rgb/")

depth_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_vkitti/virtual_world_vkitti-2/vkitti_2.0.3_depth/")

# vkitti_dataset = vkittiDataset(imgs_root, depth_root, get_transform)

# vkitti_dataset.__getitem__(1000)

data_loader_train, data_loader_val = get_dataloaders(1, imgs_root, depth_root, split=True, val_size=0.20, n_samples=None)

print(len(data_loader_train), len(data_loader_val))
# data_depth_velodyne_root = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/data_depth_velodyne/train/")
# data_depth_annotated_root = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/data_depth_annotated/train/")

# calib_velo2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/calib_velo_to_cam.txt")
# calib_cam2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/calib_cam_to_cam.txt")

# kitti_data_loader = get_dataloaders(batch_size=1, imgs_root=imgs_root,
#                                     data_depth_velodyne_root=data_depth_velodyne_root, data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam)


# iterator = iter(kitti_data_loader)

# # (img, imPts, lidar_fov, mask, sparse_depth), gt_img = next(iterator)

# img, imPts, lidar_fov, mask, sparse_depth, k_nn_indices, gt_img = next(iterator)

# # print(img[0].shape, imPts[0].shape, lidar_fov[0].shape, gt_img[0].shape)

# # img, imPts, lidar_fov, mask, sparse_depth = data_tuple[0]


# for inputs in kitti_data_loader:

#     img, imPts, lidar_fov, mask, sparse_depth, k_nn_indices, gt_img = inputs


#     print(img[0].shape, imPts[0].shape, lidar_fov[0].shape, mask[0].shape, sparse_depth[0].shape, k_nn_indices[0].shape, gt_img[0].shape)
