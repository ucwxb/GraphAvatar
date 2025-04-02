import torch
import lpips

class LPIPSLoss(torch.nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.percep_module=lpips.LPIPS(net="vgg")
        self.percep_module.to(device="cuda").eval()

    def forward(self, render, gt):
        return torch.mean(self.percep_module.forward(render, gt))