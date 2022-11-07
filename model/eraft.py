import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from model.utils import coords_grid, upflow8
from argparse import Namespace
from utils.image_utils import ImagePadder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, config, n_first_channels):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()
        self.args = args
        # ImagePadder : image downsize할 때 (15.5 와같은 이상한 사이즈가 되는 것을 막도록 하는 과정)
        self.image_padder = ImagePadder(min_size=32)
        # subtype : warm-start 
        self.subtype = config['subtype'].lower()

        assert (self.subtype == 'standard' or self.subtype == 'warm_start')

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                    n_first_channels=n_first_channels)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
                                    n_first_channels=n_first_channels)
        # fnet & cnet & current flow 를 기반으로 delta_flow 계산하는 block 
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        # ( 1, 15, 256, 256 )
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def bilinear_interpolation_kernel(self, last_flow, coords1):
        # g(x_i-1) = x_(i-1) + F_(i-1 -> i)(x_i-1)
        g_prev = coords1 + last_flow
        g_x_prev = g_prev[0, 0, :, :] # ( 32, 32 )
        g_y_prev = g_prev[0, 1, :, :] # ( 32, 32 )

        flow_init = torch.zeros((1, 2, len(coords1[0][0]), len(coords1[0][0])))
        
        for i in range(0, len(coords1[0][0])):
            for j in range(0, len(coords1[0][0])):
                ## Process : (0,0) -> ( 1, 0 ) -> ... -> ( 31, 0 ) // (1, 0)) -> ... -> ( 31, 1 )
                zero = torch.zeros(1, 32, 32).cuda()
                a_x = 1 - (coords1[0, 0, i, j] - g_x_prev).abs()
                a_x = a_x.unsqueeze(axis=0) # ( 1, 32, 32 )
                a_y = 1 - (coords1[0, 1, i, j] - g_y_prev).abs()
                a_y = a_y.unsqueeze(axis=0) # ( 1, 32, 32 )
                
                a_x = torch.cat([zero, a_x], dim=0) # ( 2, 32, 32 )
                a_y = torch.cat([zero, a_y], dim=0)
                
                x_kernel = a_x.max(dim=0)[0]
                y_kernel = a_y.max(dim=0)[0] # (32, 32)
                
                kernel = torch.mul(x_kernel, y_kernel) # 원소별 곱 
                kernel = kernel.unsqueeze(axis=0) # ( 1, 32, 32 )
                kernel = torch.cat([kernel, kernel], dim=0) # ( 2, 32, 32 )
                kernel = kernel.unsqueeze(axis=0) # ( 1, 2, 32, 32 )
                
                if (torch.sum(kernel)) == 0:
                    flow_init[0, 0, i, j] = last_flow[0, 0, i, j]
                    flow_init[0, 1, i, j] = last_flow[0, 1, i, j]
                else:
                    # last_flow : ( 1, 2, 32, 32 )
                    flow_init_x = torch.sum(torch.mul(kernel[0, 0, :, :], last_flow[0, 0, :, :])) / torch.sum(kernel[0, 0, :, :])
                    flow_init_y = torch.sum(torch.mul(kernel[0, 1, :, :], last_flow[0, 1, :, :])) / torch.sum(kernel[0, 1, :, :])
                
                    #flow_init_xy = np.sum(kernel * last_flow) / np.sum(kernel)
                    flow_init[0, 0, i, j] = flow_init_x
                    flow_init[0, 1, i, j] = flow_init_y

        
        return flow_init

    def forward(self, image1, image2, iters=12, flow_init = None, upsample=True):
        """ Estimate optical flow between pair of frames """
        # Pad Image (for flawless up&downsampling)
        image1 = self.image_padder.pad(image1)
        image2 = self.image_padder.pad(image2)

        # contiguous : 새로운 메모리 공간에 데이터 복사 ( 다른 주소값 가짐 )
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        # 128 
        hdim = self.hidden_dim
        cdim = self.context_dim 

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        # fmap 2개를 입력으로 받아 correlation volume 생성 
        # radius : 4
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if (self.subtype == 'standard' or self.subtype == 'warm_start'):
                cnet = self.cnet(image2)
            else:
                raise Exception

            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        # corrds0.shape = ( 1, 2, 32, 32 )
        coords0, coords1 = self.initialize_flow(image1)
        
        # For Warm-Start
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))

        return coords1 - coords0, flow_predictions
