import os
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.mesh_utils import extract_graph
import torch
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    seg_mask: np.array
    width: int
    height: int
    exp: np.array
    time: float
    vertice_feature: np.array
    edge_index: np.array
    valid_depth_mask: np.array
    depth: np.array

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    # nerf_normalization: dict
    exp_max: np.array
    exp_min: np.array

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", split="train", debug=False, use_retrack=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focalx = contents["fl_x"]
        focaly = contents["fl_y"]
        frames = contents["frames"]
        # if "train" in transformsfile:
        #     frames.extend(json.load(open(os.path.join(path, transformsfile.replace("train", "test"))))["frames"])
        # if len(frames) >= 2000:
        #     frames = frames[-2000:]
        test_img = os.path.join(path, frames[0]["file_path"])
        test_img = Image.open(test_img)
        (w,h) = test_img.size
        FovX = focal2fov(focalx, w)
        FovY = focal2fov(focaly, h)

        time_ori = list(range(0,len(os.listdir(os.path.join(path, "images")))))
        if debug:
            frames = frames[0:1]
        time_ori = np.array(time_ori)/len(frames)
        time_ori = time_ori.astype(np.float32)
        for idx, frame in enumerate(tqdm(frames)):
            frame_idx = int(os.path.basename(frame["file_path"]).split(".")[0])
            if use_retrack:
                retrack_path = os.path.join(path.replace("insta-dataset", "tracker"), "checkpoint", "%05d.frame"%(frame_idx))
                mesh_path = os.path.join(path.replace("insta-dataset", "tracker"), "mesh", "%05d.ply"%(frame_idx))
                vertice_feature, edge_index = extract_graph(mesh_path)
                retrack_payload = torch.load(retrack_path)
                flame_params = retrack_payload["flame"]

                oepncv = retrack_payload['opencv']
                w2cR = oepncv['R'][0]
                w2cT = oepncv['t'][0]
                R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
                T = w2cT

                orig_w, orig_h = retrack_payload['img_size']
                K = retrack_payload['opencv']['K'][0] # (3, 3)
                fl_x = K[0, 0] # 
                fl_y = K[1, 1] # 
                FovY = focal2fov(fl_y, orig_h)
                FovX = focal2fov(fl_x, orig_w)

                exp = np.concatenate([flame_params['exp'], flame_params['eyes'], flame_params['eyelids'], flame_params['jaw']], axis=1, dtype=np.float32).reshape(-1)
                
                depth = np.array(Image.open(os.path.join(path.replace("insta-dataset", "tracker"), "depth", "%05d.png"%(frame_idx)))).astype(np.float32)
                valid_depth_mask = depth > 0
            else:
                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                # c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
                exp = np.array(np.loadtxt(os.path.join(path, frame['exp_path'])).reshape(-1), dtype=np.float32)

                mesh_path = os.path.join(path, frame["mesh_path"])
                vertice_feature, edge_index = extract_graph(mesh_path)

                depth = np.array(Image.open(os.path.join(path, frame["depth_path"]))).astype(np.float32)
                valid_depth_mask = depth > 0

            image_path = os.path.join(path, frame["file_path"])
            image_name = os.path.basename(frame["file_path"])
            image = Image.open(image_path)

            # seg_path = os.path.join(path, frame["seg_mask_path"])
            # seg_image = np.array(Image.open(seg_path))
            # seg_mask = ((seg_image[:,:,0] > 90) & (seg_image[:,:,0] <= 250)) | (seg_image[:,:,0] == 80)
            seg_mask = np.array(Image.open(image_path))[:,:,-1] / 255.0


            # scale_depth = (depth[valid_depth_mask] - np.min(depth[valid_depth_mask])) / (np.max(depth[valid_depth_mask]) - np.min(depth[valid_depth_mask]))
            # depth[valid_depth_mask] = scale_depth
            time = time_ori[frame_idx]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, seg_mask=seg_mask,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], exp=exp, time=time,
                            vertice_feature=vertice_feature, edge_index=edge_index, valid_depth_mask=valid_depth_mask, depth=depth))
            
    return cam_infos

def readInstaInfo(path, white_background, eval, extension=".png",debug=False,use_retrack=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, debug=debug, use_retrack=use_retrack)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, debug=debug, use_retrack=use_retrack)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    exp_matrix = np.array([cam_info.exp for cam_info in train_cam_infos])
    exp_max = np.max(exp_matrix, axis=0)
    exp_min = np.min(exp_matrix, axis=0)
    scene_info = SceneInfo(
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            exp_max=exp_max,
                            exp_min=exp_min)
    return scene_info