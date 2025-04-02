import os
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
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
        focalx = contents["fx"]
        focaly = contents["fy"]
        h = contents["h"]
        w = contents["w"]
        FovX = focal2fov(focalx, w)
        FovY = focal2fov(focaly, h)

        frames = contents["frames"]
        time_ori = list(range(0,len(frames)))
        time_ori = np.array(time_ori)/len(frames)
        time_ori = time_ori.astype(np.float32)
        if split == "train":
            frames = frames[0:-500]
            if debug:
                frames = frames[0:1]
        elif split == "val":
            frames = frames[-500::70]
            if debug:
                frames = frames[0:10]
        elif split == "test":
            frames = frames[-500:]
            if debug:
                frames = frames[0:1]

        for idx, frame in enumerate(tqdm(frames)):
            cam_name =  "%05d.png"%(int(frame['img_id']))
            frame_ind = int(frame['img_id'])
            retrack_path = os.path.join(path.replace("nbs-dataset", "tracker"), "checkpoint", "%05d.frame"%(frame_ind))
            mesh_path = os.path.join(path.replace("nbs-dataset", "tracker"), "mesh", "%05d.ply"%(frame_ind))
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
            
            depth = np.array(Image.open(os.path.join(path.replace("nbs-dataset", "tracker"), "depth", "%05d.png"%(frame_ind)))).astype(np.float32)
            valid_depth_mask = depth > 0

            image_path = os.path.join(path, "head_neck_imgs", cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            seg_path = os.path.join(path, "head_neck_mask", "%05d.png"%(int(frame['img_id'])))
            seg_image = np.array(Image.open(seg_path))
            seg_mask = (seg_image==255).astype(np.uint8)

            time = time_ori[frame_ind]

            cam_infos.append(CameraInfo(uid=frame_ind, R=R, T=T, FovY=FovY, FovX=FovX, image=image, seg_mask=seg_mask, 
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], exp=exp, time=time,
                            vertice_feature=vertice_feature, edge_index=edge_index, valid_depth_mask=valid_depth_mask, depth=depth))
            
    return cam_infos

def readBlendshapeInfo(path, white_background, eval, extension=".jpg",debug=False,use_retrack=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension, split="train", debug=debug, use_retrack=use_retrack)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension, split="test", debug=debug, use_retrack=use_retrack)
    
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