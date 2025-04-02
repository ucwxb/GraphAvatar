import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class ScaffoldNet(nn.Module):
    def __init__(self, 
                anchor_num: int,
                feat_dim: int = 32, 
                n_offsets: int = 5, 
                use_feat_bank : bool = False, \
                z_dim: int = 0, \
        ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.use_feat_bank = use_feat_bank

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+z_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+z_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+z_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.offsets = nn.Parameter(torch.zeros((anchor_num, self.n_offsets, 3)).float().cuda().requires_grad_(True))
        self.anchors_feat = nn.Parameter(torch.zeros((anchor_num, self.feat_dim)).float().cuda().requires_grad_(True))
    
    def forward(self, anchor, grid_scaling, camera_center, z_feat):
        
        feat = self.anchors_feat
        grid_offsets = self.offsets

        ## get view properties for anchor
        ob_view = anchor - camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            
            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]

        z_feat = z_feat.repeat([anchor.shape[0], 1]) # [n, z_dim]
        cat_local_view_wodist = torch.cat([feat, ob_view, z_feat], dim=1) # [N, c+3]
        # cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

        neural_opacity = self.mlp_opacity(cat_local_view_wodist)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        opacity = neural_opacity[mask]

        color = self.mlp_color(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]

        scale_rot = self.mlp_cov(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([3, 3, 3, 7, 3], dim=-1)

        scaling = scaling_repeat * torch.sigmoid(scale_rot[:,:3])
        rot = F.normalize(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat
        xyz = repeat_anchor + offsets

        return xyz, color, opacity, scaling, rot, neural_opacity, mask