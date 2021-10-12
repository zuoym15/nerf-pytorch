import numpy as np
import glob
import os
import cv2
import imageio

# import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict


# training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
#                     45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
#                     74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
#                     101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
#                     121, 122, 123, 124, 125, 126, 127, 128]

def scale_operation(images, masks, intrinsics, s):
    ht1 = images.shape[0]
    wd1 = images.shape[1]
    ht2 = int(s * ht1)
    wd2 = int(s * wd1)
    intrinsics[0, :] *= s
    intrinsics[1, :] *= s
    images = cv2.resize(images, (wd2, ht2))
    masks = cv2.resize(masks, (wd2, ht2), interpolation=cv2.INTER_NEAREST)
    # images = F.interpolate(images, [ht2, wd2], mode='bilinear', align_corners=True)
    return images, masks, intrinsics


class DTU(object):
    def __init__(self, dataset_path, scene_name, light_number, foreground_mask_path=None, world_scale=400.0, image_scale=0.25, split='train'):
        self.dataset_path = dataset_path
        # self.num_frames = num_frames
        # self.min_angle = min_angle
        # self.max_angle = max_angle
        self.light_number = light_number
        self.foreground_mask_path = foreground_mask_path
        # self.crop_size = crop_size
        # self.resize = resize
        self.scene_name = scene_name
        self.world_scale = world_scale
        self.image_scale = image_scale
        # self.precomputed_depth_path = precomputed_depth_path
        self.split = split

        self._build_dataset_index()
        self._load_poses()

        # self.pairs_provided = pairs_provided
        # if pairs_provided:
        #     self.pair_list = load_pair(os.path.join(dataset_path, 'pair.txt'))



    # def _theta_matrix(self, poses):
    #     delta_pose = np.matmul(poses[:,None], np.linalg.inv(poses[None,:]))
    #     dR = delta_pose[:, :, :3, :3]
    #     cos_theta = (np.trace(dR,axis1=2, axis2=3) - 1.0) / 2.0
    #     cos_theta = np.clip(cos_theta, -1.0, 1.0)
    #     return np.rad2deg(np.arccos(cos_theta))

    def _load_poses(self):
        if self.split == 'train': # folder structure different for train and val/test set
            pose_glob = os.path.join(self.dataset_path, "Cameras", "train", "*.txt")
        else:
            pose_glob = os.path.join(self.dataset_path, self.scene_name, "cams", "*.txt")

        extrinsics_list, intrinsics_list = [], []
        for cam_file in sorted(glob.glob(pose_glob)):
            extrinsics = np.loadtxt(cam_file, skiprows=1, max_rows=4, dtype=np.float)
            intrinsics = np.loadtxt(cam_file, skiprows=7, max_rows=3, dtype=np.float)
            
            intrinsics[0] *= self.scale_between_image_depth
            intrinsics[1] *= self.scale_between_image_depth
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

        poses = np.stack(extrinsics_list, 0)

        # convert w2c to c2w
        poses = np.linalg.inv(poses)

        transf = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1.],
        ])

        poses = poses @ transf # to opengl pose
        poses[:, :3, 3] /= self.world_scale # make the xyz range near 1

        poses = poses[:, :3, :4] # N x 3 x 4
        # print(np.var(poses[:, :3, 3], axis=0))
        intrinsics = np.stack(intrinsics_list, 0)

        self.poses = poses
        self.intrinsics = intrinsics

    def _build_dataset_index(self):
        self.dataset_index = []
        if self.split == 'train':
            image_path = os.path.join(self.dataset_path, "Rectified")
        else:
            image_path = self.dataset_path
        # depth_path = os.path.join(self.dataset_path, "Depths")
        # self.scale_between_image_depth = None
        self.scale_between_image_depth = 1.0
        self.scenes = {}
        for scene in os.listdir(image_path):
            # print(scene)
            # print(self.scene_name)
            if scene != self.scene_name:
                continue
            # if scene[-6:] == "_train": id = int(scene[4:-6])
            # else: id = int(scene[4:])
            # if not id in training_set: continue
            if self.split == 'train':
                k = self.light_number
                images = sorted(glob.glob(os.path.join(image_path, scene, "*_%d_*.png" % k)))
                scene_id = "%s_%d" % (scene, k)
            else:
                images = sorted(glob.glob(os.path.join(image_path, scene, "images", "*.jpg")))
                scene_id = scene

            self.scenes[scene_id] = images
            self.dataset_index += [(scene_id, i) for i in range(len(images))]

        # print("scale_between_image_depth", self.scale_between_image_depth)
        # print('Dataset length:', len(self.dataset_index))

    def __len__(self):
        return len(self.dataset_index)

    def load_dtu_data(self):
        scene_id, _ = self.dataset_index[0]
        image_list = self.scenes[scene_id]

        indicies = np.array(list(range(len(image_list))))

        images, depths, poses, intrinsics, masks = [], [], [], [], []
        for i in indicies:
            image = imageio.imread(image_list[i]) / 255.0

            pose = self.poses[i]
            calib = self.intrinsics[i]

            if self.foreground_mask_path is not None:
                img_name = os.path.basename(image_list[i]).split('_')[1]  # e.g. 002
                img_name = str((int(img_name) - 1)).zfill(3)

                scene_name = scene_id.split('_')[0]  # e.g. scan3
                mask_fn = os.path.join(self.foreground_mask_path, scene_name, 'mask', img_name + '.png')

                if not os.path.exists(mask_fn):
                    print('warning: mask path %s does not exist!' % mask_fn )
                    mask = np.ones_like(image[..., 0:1])  # all fg as default

                else:
                    mask = imageio.imread(mask_fn) / 255.0
                    if mask.ndim == 3:
                        mask = mask[..., 0:1]  # in case encoded as colorful image
            else:
                mask = np.ones_like(image[..., 0:1]) # all fg as default

            image, mask, calib = scale_operation(image, mask, calib, self.image_scale)

            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)
            masks.append(mask)

        masks = np.stack(masks, 0).astype(np.float32)
        images = np.stack(images, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)

        if masks.ndim==3:
            masks = masks[..., None]

        # focal = (intrinsics[0,0,0] + intrinsics[0,1,1]) / 2.
        # # print('focal_x', focal, 'focal_y', intrinsics[0,1,1])
        # H = images.shape[1]
        # W = images.shape[2]

        train_split = indicies[np.mod(np.arange(len(indicies), dtype=int), 8) != 0]
        val_split = []
        test_split = indicies[np.mod(np.arange(len(indicies), dtype=int), 8) == 0]

        i_split = [train_split, val_split, test_split]

        render_poses = poses[test_split]

        return images, masks, poses, render_poses, intrinsics[0], i_split

if __name__ == '__main__':
    gpuargs = {'num_workers': 4, 'drop_last' : True, 'shuffle': True, 'pin_memory': True}
    train_dataset = DTU('/n/fs/pvl-mvs/DTU_HR/train')
    # train_loader = DataLoader(train_dataset, batch_size=4, **gpuargs)

    # for (images, depths, poses, intrinsics) in train_loader:
    #     print(images.shape)

