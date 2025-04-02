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

import os
import time
import random
import shutil
import torch
import numpy as np
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state,inverse_sigmoid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vgg_loss import LPIPSLoss
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.dataset_blendshape import storePly
from utils.logger import Logger
import torchvision
import torch.nn.functional as F
from utils.sh_utils import eval_sh
from torchmetrics import PearsonCorrCoef
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def load_warmup_gaussians(warmup_load, max_sh_degree, is_act=True):
    plydata = PlyData.read(warmup_load)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])

    features_rest = features_rest.reshape((features_rest.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    features_dc = torch.from_numpy(features_dc).cuda().transpose(1,2)
    features_rest = torch.from_numpy(features_rest).cuda().transpose(1,2)
    shs = torch.cat((features_dc, features_rest), dim=1)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    means3D = torch.from_numpy(xyz).cuda()
    scales = torch.from_numpy(scales).cuda()
    rotations = torch.from_numpy(rots).cuda()
    opacity = torch.from_numpy(opacities).cuda()

    if is_act:
        scales = torch.exp(scales)
        rotations = torch.nn.functional.normalize(rotations)
        opacity = torch.sigmoid(opacity)
    else:
        scales = scales
        rotations = rotations
        opacity = opacity
    
    return means3D, scales, rotations, opacity, shs

def training(args, dataset, opt, pipe, test_and_save_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset) if not args.render_only else None
    print(args)
    gaussians = GaussianModel(dataset.sh_degree, \
                                exp_num=args.exp_num, \
                                exp_mlp_dim=args.exp_mlp_dim, \
                                bs_template_path=args.bs_template_path, \
                                opt_params = args.wo_opt_params, \
                                post_process = args.post_process, \
                                z=args.z_dim, \
                                downsampling_factors = args.downsampling_factors, \
                                polygon_order = args.polygon_order, \
                                num_conv_filters = args.num_conv_filters, \
                                n_layers = args.n_layers, \
                                enable_scaffold = args.enable_scaffold, \
                                feat_dim = args.scaffold_feat_dim, \
                                n_offsets = args.scaffold_n_offsets, \
                                use_feat_bank = args.scaffold_use_feat_bank, \
                                train_rgb = args.train_rgb, \
                            )

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, map_location="cpu")
        gaussians.restore(model_params, opt)

    if "white" in dataset.white_background:
        bg_color = np.array([1,1,1])
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    elif "black" in dataset.white_background:
        bg_color = np.array([0,0,0])
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    elif dataset.white_background.endswith(".jpg") or dataset.white_background.endswith(".png"):
        bg_color = np.array(Image.open(os.path.join(dataset.source_path,dataset.white_background))) / 255.0
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda").permute(2,0,1)

    if args.render_only:
        metrics_dict = {
            "test":{
                "l2":0.0,
                "psnr":0.0,
                "ssim":0.0,
                "lpips":0.0,
                "time":0.0
            }
        }
        
        test_reder_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), "render_test")
        os.makedirs(test_reder_path, exist_ok=True)

        gaussians.set_eval_all()
        with torch.no_grad():
            for idx, viewpoint in enumerate(tqdm(scene.getTestCameras())):
                gt_image = viewpoint.original_image.cuda()
                gt_alpha_mask = viewpoint.gt_alpha_mask.cuda()
                bg = torch.rand_like(gt_image, device="cuda") if opt.random_background else background
                if len(bg.shape) == 1:
                    bg = bg.unsqueeze(-1).unsqueeze(-1).repeat((1, gt_image.shape[1], gt_image.shape[2]))
                s_time = time.time()
                render_pkg = render(viewpoint, gaussians, pipe, bg, vertice_feature = viewpoint.vertice_feature, edge_index = viewpoint.edge_index)
                metrics_dict["test"]["time"] += time.time() - s_time
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                torchvision.utils.save_image(image, os.path.join(test_reder_path, viewpoint.image_name))

                gt_image = gt_image * gt_alpha_mask[None,:,:] + (bg * (1 - gt_alpha_mask[None,:,:]))
                metrics_dict["test"]["l2"] += l2_loss(image, gt_image).mean().double()
                metrics_dict["test"]["psnr"] += psnr(image, gt_image).mean().double()
                metrics_dict["test"]["ssim"] += ssim(image, gt_image).mean().double()
                metrics_dict["test"]["lpips"] += lpips(image, gt_image).mean().double()

                # gt_alpha_mask = viewpoint.gt_alpha_mask.cuda()
                # depth_map = render_pkg["depth_map"]
                # valid_depth = (depth_map != 0) & (gt_alpha_mask.bool())
                # scale_depth = (depth_map[valid_depth] - depth_map[valid_depth].min()) / (depth_map[valid_depth].max() - depth_map[valid_depth].min())
                # depth_map[valid_depth] = scale_depth
                # if len(depth_map.shape) == 2:
                #     depth_map = depth_map.unsqueeze(0)
                # depth_map[~(gt_alpha_mask.unsqueeze(0).bool())] = 0
                # torchvision.utils.save_image(depth_map, "depth.png")
                

        metrics_dict["test"]["l2"] /= len(scene.getTestCameras())
        metrics_dict["test"]["psnr"] /= len(scene.getTestCameras())
        metrics_dict["test"]["ssim"] /= len(scene.getTestCameras())
        metrics_dict["test"]["lpips"] /= len(scene.getTestCameras())
        metrics_dict["test"]["time"] /= len(scene.getTestCameras())
        print("Rendering test path: ", test_reder_path)
        print("\n[EVALUATING TEST]: L2 {} PSNR {} SSIM {} LPIPS {} TIME {}".format(metrics_dict["test"]["l2"], metrics_dict["test"]["psnr"], \
                                                                                metrics_dict["test"]["ssim"], metrics_dict["test"]["lpips"], metrics_dict["test"]["time"]))
        return

    

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    vgg_loss = LPIPSLoss().cuda().eval()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    is_act = True
    psnr_best = 0.0
    means3D_t, scales_t, rotations_t, opacity_t, shs_t = load_warmup_gaussians(opt.warmup_load, dataset.sh_degree, is_act=is_act)

    pearson_loss = PearsonCorrCoef().cuda()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        lr_cur = gaussians.update_learning_rate(iteration)
        gaussians.set_train_all()
        if iteration < opt.warmup_iteration:

            means3D, scales, rotations, opacity, shs, cam_coff = gaussians.forward_graph(viewpoint_cam.vertice_feature, \
                                                                                        viewpoint_cam.edge_index, \
                                                                                        viewpoint_cam.exp, \
                                                                                        is_act=is_act, \
                                                                                        time=viewpoint_cam.time)
            if gaussians.opt_params:
                viewpoint_cam.update_param(cam_coff)

            loss_pos = l1_loss(means3D, means3D_t)
            loss_scale = l1_loss(scales, scales_t)
            loss_rot = l1_loss(rotations, rotations_t)
            loss_opa = l1_loss(opacity, opacity_t)

            if args.train_rgb:
                shs_view = shs_t.clone().transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                dir_pp = (means3D_t - viewpoint_cam.camera_center.detach().repeat(shs_t.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                color_t = torch.clamp_min(sh2rgb + 0.5, 0.0)
                color_t = torch.clamp_max(color_t, 1.0)
                loss_shs = l1_loss(shs, color_t)
            else:
                loss_shs = l1_loss(shs, shs_t)
            loss = loss_pos + loss_scale + loss_rot + loss_opa + loss_shs
            Ll1 = loss

        else:
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            # merge bg and gt_image

            bg = torch.rand_like(gt_image, device="cuda") if opt.random_background else background

            if len(bg.shape) == 1:
                bg = bg.unsqueeze(-1).unsqueeze(-1).repeat((1, gt_image.shape[1], gt_image.shape[2]))

            debug_dict = {}
            debug_dict["means3D"] = means3D_t.float()
            debug_dict["scales"] = scales_t.float()
            debug_dict["rotations"] = rotations_t.float()
            debug_dict["opacity"] = opacity_t.float()
            debug_dict["shs"] = shs_t.float()

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, vertice_feature = viewpoint_cam.vertice_feature, edge_index = viewpoint_cam.edge_index, debug_dict=debug_dict)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image_pre = render_pkg["render_pre"]
            depth_map = render_pkg["depth_map"]
            weight_map = render_pkg["weight_map"]

            gt_image = gt_image * gt_alpha_mask[None,:,:] + (bg * (1 - gt_alpha_mask[None,:,:]))

            Ll1 = l1_loss(image, gt_image)
            loss_pho = (1.0 - opt.lambda_dssim) * Ll1
            loss_ssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss = loss_pho + loss_ssim
            
            loss_vgg = 0.0
            if iteration > 20_000 and opt.lambda_vgg > 0.0:
                loss_vgg = opt.lambda_vgg * vgg_loss(image, gt_image)
                loss += loss_vgg
            
            loss_pre = 0.0
            if args.pre_loss:
                loss_pre =  l1_loss(image_pre, gt_image) + \
                            opt.lambda_dssim * (1.0 - ssim(image_pre, gt_image))
                if iteration > 20_000:
                    loss_pre += opt.lambda_vgg * vgg_loss(image_pre, gt_image)
                loss_pre *= args.lambda_pre_loss
                loss += loss_pre

            loss_metric_xyz = 0.0
            if args.metric_xyz:
                loss_metric_xyz = F.relu(render_pkg["means3D"][visibility_filter].norm(dim=1) - args.threshold_xyz).mean() * args.lambda_xyz
                loss += loss_metric_xyz
            
            loss_metric_scale = 0.0
            if args.metric_scale:
                loss_metric_scale = F.relu(render_pkg["scales"][visibility_filter] - args.threshold_scale).norm(dim=1).mean() * args.lambda_scale
                loss += loss_metric_scale
            
            loss_weight = 0.0
            if args.weight_loss:
                loss_weight = F.binary_cross_entropy(weight_map, gt_alpha_mask.float()) * args.lambda_weight_loss
                loss += loss_weight
            
            loss_depth = 0.0
            if args.depth_loss:
                depth_gt = viewpoint_cam.depth
                valid_depth_mask_gt = viewpoint_cam.valid_depth_mask
                scale_depth_gt = (depth_gt[valid_depth_mask_gt] - depth_gt[valid_depth_mask_gt].min()) / (depth_gt[valid_depth_mask_gt].max() - depth_gt[valid_depth_mask_gt].min())
                scale_depth_map = (depth_map[valid_depth_mask_gt] - depth_map[valid_depth_mask_gt].min()) / (depth_map[valid_depth_mask_gt].max() - depth_map[valid_depth_mask_gt].min())

                loss_depth = (1 - pearson_loss(scale_depth_gt, scale_depth_map)) * args.lambda_depth_loss
                loss += loss_depth

        loss.backward()

        iter_end.record()

        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            if iteration < opt.warmup_iteration:
                progress_bar.set_postfix({"lr": f"{lr_cur:.{4}f}", \
                                        "pos": f"{loss_pos:.{4}f}", \
                                        "scl": f"{loss_scale:.{4}f}", \
                                        "rot": f"{loss_rot:.{4}f}", \
                                        "opa": f"{loss_opa:.{4}f}", \
                                        "shs": f"{loss_shs:.{4}f}", \
                                        "tot": f"{loss:.{4}f}"
                                        })
                progress_bar.update(10)
            else:
                progress_bar.set_postfix({"lr": f"{lr_cur:.{4}f}", \
                                        "pho": f"{loss_pho:.{4}f}", \
                                        "ssim": f"{loss_ssim:.{4}f}", \
                                        "vgg": f"{loss_vgg:.{4}f}", \
                                        "lw": f"{loss_weight:.{4}f}", \
                                        "lp": f"{loss_pre:.{4}f}", \
                                        "de": f"{loss_depth:.{4}f}", \
                                        })
                progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()

        with torch.no_grad():
            # Progress bar
            debug_dict = {}
            debug_dict["means3D"] = means3D_t.float()
            debug_dict["scales"] = scales_t.float()
            debug_dict["rotations"] = rotations_t.float()
            debug_dict["opacity"] = opacity_t.float()
            debug_dict["shs"] = shs_t.float()
            # Log and save
            gaussians.set_eval_all()
            psnr_test = training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), test_and_save_iterations, scene, render, (pipe, background, debug_dict), iteration >= opt.iterations)

            if (iteration % test_and_save_iterations == 0 and psnr_test > psnr_best) or iteration >= opt.iterations:
                print("\n[ITER {}] Saving Checkpoint with psnr {}".format(iteration, psnr_test))
                psnr_best = psnr_test
                chk_path = os.path.join(scene.model_path, "chk")
                os.makedirs(chk_path, exist_ok=True)
                torch.save((gaussians.capture(), iteration), os.path.join(chk_path, "chkpnt" + str(iteration) + "_%.4f.pth"%(psnr_test)))

def prepare_output_and_logger(args):
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_path = os.path.join(args.model_path, "tb")
        os.makedirs(tb_path, exist_ok=True)
        tb_writer = SummaryWriter(tb_path)
    else:
        print("Tensorboard not available: not logging progress")

    os.makedirs(os.path.join(args.model_path, "render"), exist_ok=True)
    # copy code
    code_path = os.path.join(args.model_path, "scripts")
    os.makedirs(code_path, exist_ok = True)
    scan_path = ["./", "./arguments", "./gaussian_renderer","./gridencoder","./scene","./utils"]
    for scan_p in scan_path:
        for script in os.listdir(scan_p):
            if script.split('.')[-1] in ['sh','py']:
                dst_file = os.path.join(code_path, scan_p, os.path.basename(script))
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copyfile(os.path.join(scan_p, script), dst_file)

    sys.stdout = Logger(os.path.join(args.model_path, 'log.txt'))
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed, test_and_save_iterations, scene : Scene, renderFunc, renderArgs, save_img: bool):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration % test_and_save_iterations == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        metrics_dict = {
            "test":{
                "l2":0.0,
                "psnr":0.0,
                "ssim":0.0,
                "lpips":0.0,
                "time":0.0
            },
            "train":{
                "l2":0.0,
                "psnr":0.0,
                "ssim":0.0,
                "lpips":0.0,
                "time":0.0
            }
        }
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_alpha_mask = viewpoint.gt_alpha_mask.cuda()
                    # merge bg and gt_image

                    if len(renderArgs[1].shape) == 1:
                        bg_c = renderArgs[1].unsqueeze(-1).unsqueeze(-1).repeat((1, gt_image.shape[1], gt_image.shape[2]))
                    else:
                        bg_c = renderArgs[1]
                    time_s = time.time()
                    render_dict = renderFunc(viewpoint, scene.gaussians, pipe=renderArgs[0], bg_color=bg_c, vertice_feature = viewpoint.vertice_feature, edge_index = viewpoint.edge_index, debug_dict=renderArgs[2])
                    image = torch.clamp(render_dict["render"], 0.0, 1.0)
                    image_pre = torch.clamp(render_dict["render_pre"], 0.0, 1.0)
                    weight_map = torch.clamp(render_dict["weight_map"], 0.0, 1.0)
                    metrics_dict[config['name']]["time"] += time.time() - time_s
                    gt_image = gt_image * gt_alpha_mask[None,:,:] + (bg_c * (1 - gt_alpha_mask[None,:,:]))
                    if tb_writer and (idx < 5):
                        depth_map = render_dict["depth_map"]
                        valid_depth = depth_map != 0
                        scale_depth = (depth_map[valid_depth] - depth_map[valid_depth].min()) / (depth_map[valid_depth].max() - depth_map[valid_depth].min())
                        depth_map[valid_depth] = scale_depth
                        if len(depth_map.shape) == 2:
                            depth_map = depth_map.unsqueeze(0)

                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_pre".format(viewpoint.image_name), image_pre[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/weight_map".format(viewpoint.image_name), weight_map[None, None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth_map".format(viewpoint.image_name), depth_map[None], global_step=iteration)
                    if save_img and config['name'] == 'test':
                        torchvision.utils.save_image(image, os.path.join(scene.model_path, "render", viewpoint.image_name))
                    
                    metrics_dict[config['name']]["l2"] += l2_loss(image, gt_image).mean().double()
                    metrics_dict[config['name']]["psnr"] += psnr(image, gt_image).mean().double()
                    metrics_dict[config['name']]["ssim"] += ssim(image, gt_image).mean().double()
                    metrics_dict[config['name']]["lpips"] += lpips(image, gt_image).mean().double()

                metrics_dict[config['name']]["l2"] /= len(config['cameras'])
                metrics_dict[config['name']]["psnr"] /= len(config['cameras'])
                metrics_dict[config['name']]["ssim"] /= len(config['cameras'])
                metrics_dict[config['name']]["lpips"] /= len(config['cameras'])
                metrics_dict[config['name']]["time"] /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L2 {} PSNR {} SSIM {} LPIPS {} TIME {}".format(iteration, config['name'],\
                                                                                        metrics_dict[config['name']]["l2"], metrics_dict[config['name']]["psnr"], \
                                                                                        metrics_dict[config['name']]["ssim"], metrics_dict[config['name']]["lpips"],\
                                                                                        metrics_dict[config['name']]["time"]))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l2_loss', metrics_dict[config['name']]["l2"], iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', metrics_dict[config['name']]["psnr"], iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', metrics_dict[config['name']]["ssim"], iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', metrics_dict[config['name']]["lpips"], iteration)

        torch.cuda.empty_cache()
        return metrics_dict['test']["psnr"]

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_and_save_iterations", type=int, default=1000)
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # newly added
    parser.add_argument('--exp_num', type=int, default=46)
    parser.add_argument('--exp_mlp_dim', type=int, default=64)
    parser.add_argument('--wo_opt_params', default=True, action='store_false')
    parser.add_argument('--post_process', default=False, action='store_true')

    parser.add_argument('--metric_xyz', default=False, action='store_true')
    parser.add_argument('--metric_scale', default=False, action='store_true')

    parser.add_argument("--lambda_xyz", type=float, default=1e-2)
    parser.add_argument("--lambda_scale", type=float, default=1.0)

    parser.add_argument("--threshold_xyz", type=float, default=1.0)
    parser.add_argument("--threshold_scale", type=float, default=0.6)

    parser.add_argument('--weight_loss', default=False, action='store_true')
    parser.add_argument('--lambda_weight_loss', type=float, default=1.0)
    
    parser.add_argument('--depth_loss', default=False, action='store_true')
    parser.add_argument('--lambda_depth_loss', type=float, default=1.0)

    parser.add_argument('--pre_loss', default=False, action='store_true')
    parser.add_argument('--lambda_pre_loss', type=float, default=1.0)

    parser.add_argument('--downsampling_factors', type=int, nargs='+', default=[4,8])
    parser.add_argument('--polygon_order', type=int, nargs='+', default=[6, 6, 6])
    parser.add_argument('--num_conv_filters', type=int, nargs='+', default=[16, 16, 16])
    parser.add_argument('--n_layers', type=int, default=2)

    parser.add_argument("--z_dim", type=int, default=8)

    # scaffold
    parser.add_argument('--enable_scaffold', default=False, action='store_true')
    parser.add_argument('--scaffold_feat_dim', type=int, default=32)
    parser.add_argument('--scaffold_n_offsets', type=int, default=5)
    parser.add_argument('--scaffold_use_feat_bank', default=False, action='store_true')

    parser.add_argument('--train_rgb', default=False, action='store_true')

    parser.add_argument('--render_only', default=False, action='store_true')
    
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_and_save_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
