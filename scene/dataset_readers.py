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

from scene.dataset_blendshape import readBlendshapeInfo
from scene.dataset_insta import readInstaInfo


sceneLoadTypeCallbacks = {
    "NeRFBlendshape": readBlendshapeInfo,
    "Insta": readInstaInfo,    
}