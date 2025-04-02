#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getProjectionMatrix, getWorld2View2_torch

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", exp=None, exp_max=None, exp_min=None, time=0.0,
                 vertice_feature=None,
                 edge_index=None,
                 depth=None,
                 valid_depth_mask=None,
                ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = torch.from_numpy(R).cuda().float()
        self.T = torch.from_numpy(T).cuda().float()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = torch.tensor(time).cuda()
        self.exp = torch.tensor(exp).cuda() if exp is not None else None
        self.exp_max = torch.tensor(exp_max).cuda() if exp_max is not None else None
        self.exp_min = torch.tensor(exp_min).cuda() if exp_min is not None else None
        self.vertice_feature = torch.tensor(vertice_feature).cuda() if vertice_feature is not None else None
        self.edge_index = torch.tensor(edge_index).cuda() if edge_index is not None else None
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.gt_alpha_mask = torch.tensor(gt_alpha_mask).to(self.data_device).to(torch.uint8)
        self.depth = torch.tensor(depth).to(self.data_device) if depth is not None else None
        self.valid_depth_mask = torch.tensor(valid_depth_mask).to(self.data_device) if valid_depth_mask is not None else None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = torch.from_numpy(trans).cuda().float()
        self.scale = torch.tensor(scale).cuda()

        self.world_view_transform = getWorld2View2_torch(self.R, self.T, self.trans, self.scale).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.full_proj_transform = torch.matmul(self.world_view_transform, self.projection_matrix)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update_param(self, cam_coff):
        R_coff = cam_coff[:3,:3]
        T_coff = cam_coff[:3,3]
        self.R = self.R @ R_coff
        self.T = self.T + T_coff

        self.world_view_transform = getWorld2View2_torch(self.R, self.T, self.trans, self.scale).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

