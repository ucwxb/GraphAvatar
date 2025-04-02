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
# import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_milestone_lr_func
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh
from scene.GUNet import GUNet
from scene.GGO import GGO
from scene.DEnhancer import DEnhancer
from scene.ScaffoldNet import ScaffoldNet

# We make an exception on snake case conventions because SO3 != so3.
def exp_map_SO3xR3(tangent_vector):  # pylint: disable=invalid-name
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.color_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, \
                exp_num : int = 46, \
                exp_mlp_dim : int = 64, \
                bound: int=1, \
                bs_template_path="", \
                opt_params=True, \
                post_process=True,\
                z = 8, \
                downsampling_factors = [4,8], \
                polygon_order = [6, 6, 6], \
                num_conv_filters = [16, 16, 16], \
                n_layers = 2, \
                # scaffold
                enable_scaffold = False, \
                feat_dim: int=32, 
                n_offsets: int=5, 
                use_feat_bank : bool = False, \
                train_rgb: bool = False, \
                ):
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        self.exp_num = exp_num
        self.exp_mlp_dim = exp_mlp_dim
        self.opt_params = opt_params
        self.post_process = post_process

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.use_feat_bank = use_feat_bank
        self.enable_scaffold = enable_scaffold
        self.train_rgb = train_rgb

        self.z = z
        self.geo_unet = GUNet(3 + 3, 3 + 3 + 4, template_mesh=bs_template_path,\
                            downsampling_factors = downsampling_factors, \
                            polygon_order = polygon_order, \
                            z = self.z, \
                            num_conv_filters = num_conv_filters, \
                            n_layers = n_layers,
                            exp_num=self.exp_num
                            ).cuda()
        if self.train_rgb:
            rgb_dim = 3
        else:
            rgb_dim = 3 * (self.max_sh_degree + 1) ** 2
        self.app_unet = GUNet(3 + 3, rgb_dim + 1, template_mesh=bs_template_path, \
                            downsampling_factors = downsampling_factors, \
                            polygon_order = polygon_order, \
                            z = self.z, \
                            num_conv_filters = num_conv_filters, \
                            n_layers = n_layers, \
                            exp_num=self.exp_num
                            ).cuda()

        self.ggo = GGO(self.exp_num, z=self.geo_unet.z+self.app_unet.z).cuda()

        self.denhancer = DEnhancer(n_in_colors=3, scale=1, num_feat=16, num_block=3, num_grow_ch=8, num_cond=1, dswise=False).cuda()

        self.scaffold_net = ScaffoldNet(anchor_num=self.geo_unet.graph_vertex_num, 
                                        feat_dim=self.feat_dim, n_offsets=self.n_offsets, 
                                        use_feat_bank=self.use_feat_bank,
                                        z_dim=self.exp_num).cuda()

        # print("Geo_unet Params: ", sum(p.numel() for p in self.geo_unet.parameters() if p.requires_grad))
        # print("App_unet Params: ", sum(p.numel() for p in self.app_unet.parameters() if p.requires_grad))
        # print("Graph_guided_optimization Params: ", sum(p.numel() for p in self.ggo.parameters() if p.requires_grad))
        # print("Post_process_net Params: ", sum(p.numel() for p in self.denhancer.parameters() if p.requires_grad))
        # print("Scaffold_net Params: ", sum(p.numel() for p in self.scaffold_net.parameters() if p.requires_grad))

        self.optimizer = None
        self.setup_functions()

    def set_train_all(self):
        self.geo_unet.train()
        self.app_unet.train()
        self.ggo.train()
        self.denhancer.train()
        self.scaffold_net.train()
    
    def set_eval_all(self):
        self.geo_unet.eval()
        self.app_unet.eval()
        self.ggo.eval()
        self.denhancer.eval()
        self.scaffold_net.eval()

    def capture(self):
        return (
            self.geo_unet.state_dict(),
            self.app_unet.state_dict(),
            self.ggo.state_dict(),
            self.denhancer.state_dict(),
            self.scaffold_net.state_dict(),
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (geo_unet_dict,
        app_unet_dict,
        ggo_dict,
        post_process_net_dict,
        scaffold_net_dict,
        opt_dict,
        ) = model_args
        self.training_setup(training_args)
        self.geo_unet.load_state_dict(geo_unet_dict)
        self.app_unet.load_state_dict(app_unet_dict)
        self.ggo.load_state_dict(ggo_dict)
        self.denhancer.load_state_dict(post_process_net_dict)
        self.scaffold_net.load_state_dict(scaffold_net_dict)

        self.optimizer.load_state_dict(opt_dict)

    def forward_GGO(self, time, geo_feat, app_feat):
        if len(time.shape) == 0:
            time = time.unsqueeze(0).unsqueeze(0)
        res = self.ggo(time, geo_feat, app_feat)
        exp_coff = res[:self.exp_num]
        cam_coff = res[self.exp_num:self.exp_num+6]
        cam_coff = exp_map_SO3xR3(cam_coff.unsqueeze(0)).squeeze(0)
        return exp_coff, cam_coff

    def forward_post_process(self, image, depth_map):
        if self.post_process:
            if len(image.shape) == 3: image = image.unsqueeze(0)
            if len(depth_map.shape) == 2: depth_map = depth_map.unsqueeze(0)
            image_post = self.denhancer(image, depth_map)
            if len(image_post.shape) == 4: image_post = image_post.squeeze(0)
        else:
            image_post = image
        return image_post

    def forward_graph(self, vertice_feature, edge_index, exp, time=None, is_act=True, camera_center=None):

        position = vertice_feature[:,:3]

        geo_feat_enc = self.geo_unet.forward_encoder(vertice_feature, edge_index)
        app_feat = self.app_unet.forward_encoder(vertice_feature, edge_index)

        cam_coff = None
        if time is not None and self.opt_params:
            exp_coff, cam_coff = self.forward_GGO(time, geo_feat_enc, app_feat)
            exp = exp + exp_coff

        geo_feat = self.geo_unet.forward_decoder(geo_feat_enc, edge_index, concat_exp=True, exp=exp).squeeze(0)
        app_feat = self.app_unet.forward_decoder(app_feat, edge_index, concat_exp=True, exp=exp).squeeze(0)

        position_offset = geo_feat[:, :3] # bound TODO
        position = position + position_offset
        if self.train_rgb:
            color = self.color_activation(app_feat[:,:-1].reshape(-1, 3))
        else:
            color = app_feat[:,:-1].reshape(-1, (self.max_sh_degree + 1) ** 2, 3)

        if is_act:
            scale = self.scaling_activation(geo_feat[:, 3:6])
            rotation = self.rotation_activation(geo_feat[:, 6:10])
            opacity = self.opacity_activation(app_feat[:, -1:])
        else:
            scale = geo_feat[:, 3:6]
            rotation = geo_feat[:, 6:10]
            opacity = app_feat[:, -1:]

        if self.enable_scaffold and camera_center is not None:
            xyz_, color_, opacity_, scaling_, rot_, neural_opacity, mask = self.scaffold_net(position, scale, camera_center.detach(), exp.unsqueeze(0))

            if not self.train_rgb:
                shs_view = color.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
                dir_pp = (position - camera_center.detach().repeat(color.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                color = torch.clamp_min(sh2rgb + 0.5, 0.0)
                color = torch.clamp_max(color, 1.0)

            position = torch.cat([position, xyz_], dim=0)
            scale = torch.cat([scale, scaling_], dim=0)
            rotation = torch.cat([rotation, rot_], dim=0)
            opacity = torch.cat([opacity, opacity_], dim=0)
            color = torch.cat([color, color_], dim=0)

        return position, scale, rotation, opacity, color, cam_coff

    def training_setup(self, training_args):
        l = [
            {'params': self.geo_unet.parameters(), 'lr': training_args.lr_rates[0], "name": "geo_unet"},
            {'params': self.app_unet.parameters(), 'lr': training_args.lr_rates[0], "name": "app_unet"},
            {'params': self.ggo.parameters(), 'lr': training_args.lr_rates[0], "name": "GGO"},
            {'params': self.denhancer.parameters(), 'lr': training_args.lr_rates[0] * 2.0, "name": "post_process_net"},
            {'params': self.scaffold_net.parameters(), 'lr': training_args.lr_rates[0] * 2.0, "name": "scaffold_net"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, weight_decay=0.0005)

        self.scheduler_args = get_milestone_lr_func(lr_rates=training_args.lr_rates, milestones=training_args.milestones)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr = 0.0
        lr_return = 0.0
        for param_group in self.optimizer.param_groups:
            lr = self.scheduler_args(iteration)
            if param_group["name"] == "scaffold_net" or param_group["name"] == "post_process_net":
                lr = lr * 2.0
            else:
                lr_return = lr
            param_group['lr'] = lr
        return lr_return