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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import time

def render(viewpoint_camera, model : GaussianModel, pipe, bg_color : torch.Tensor, \
        scaling_modifier = 1.0, override_color = None, \
        vertice_feature = None, edge_index = None, \
        debug_dict={}):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # if "means3D" in debug_dict.keys():
    #     means3D = debug_dict["means3D"]
    #     scales = debug_dict["scales"]
    #     rotations = debug_dict["rotations"]
    #     opacity = debug_dict["opacity"]
    #     shs = debug_dict["shs"]
    # else:

    means3D, scales, rotations, opacity, color, cam_coff = model.forward_graph(vertice_feature=vertice_feature, \
                                                                            edge_index=edge_index, \
                                                                            exp = viewpoint_camera.exp, \
                                                                            time=viewpoint_camera.time, \
                                                                            camera_center=viewpoint_camera.camera_center)

    if model.opt_params:
        viewpoint_camera.update_param(cam_coff)

    # print("point num:", means3D.shape) # 5254, 3

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image_pre, radii, depth_map, weight_map = rasterizer(
        means3D = means3D, # [n, 3]
        means2D = means2D, # [n, 3]
        shs = color if not (model.enable_scaffold or model.train_rgb) else None,
        colors_precomp = color if (model.enable_scaffold or model.train_rgb) else None,
        opacities = opacity, # [n, 1]
        scales = scales,    # [n, 3]
        rotations = rotations, # [n, 4]
        cov3D_precomp = None)
    rendered_image = model.forward_post_process(rendered_image_pre, depth_map)

    # print(rendered_image.isnan().sum())
    # print(shs.isnan().sum())
    # print(opacity.isnan().sum())
    # print(scales.isnan().sum())
    # print(opacity.max(), opacity.min())
    # print(opacity.max(), opacity.min())
    # print(scales.max(), scales.min())
    # print(rotations.max(), rotations.min())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_pre": rendered_image_pre,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth_map": depth_map,
            "weight_map": weight_map,
            "means3D":means3D, 
            "scales":scales, 
            "rotations":rotations,
            "opacity":opacity,
            "color":color
            }
