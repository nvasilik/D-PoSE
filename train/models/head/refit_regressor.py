import math
import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn as nn
from ...core.config import SMPL_MEAN_PARAMS
from ...core.constants import NUM_JOINTS_SMPLX
from ...utils.geometry import rot6d_to_rotmat
import numpy as np

#Taken from ReFit Repository https://github.com/yufu-wang/ReFit
class MultiLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['n_head', 'in_features', 'out_features']
    n_head: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, n_head: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.empty((n_head, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(n_head, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        out = torch.einsum('kij, bkj -> bki', self.weight, input)
        if self.bias is not None:
            out += self.bias
        return out.contiguous()

    def extra_repr(self) -> str:
        return 'n_head={}, in_features={}, out_features={}, bias={}'.format(
            self.n_head, self.in_features, self.out_features, self.bias is not None
        )

class Regressor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, num_layer=1, pose_dim=6):
        super(Regressor, self).__init__()
        input_dim = input_dim + 3
        pose_input = (720+48 +3)+6
        shape_input = (720+48+3)*22+11
        cam_input = (720+48+3)*22+3
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:NUM_JOINTS_SMPLX*6]).unsqueeze(0)
        init_shape_ = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_shape = torch.cat((init_shape_, torch.zeros((1,1))),-1)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.p = self._make_multilinear(num_layer, 22, pose_input, hidden_dim)
        self.s = self._make_linear(num_layer, shape_input, hidden_dim)
        self.c = self._make_linear(num_layer, cam_input, hidden_dim)

        self.decpose = MultiLinear(22, hidden_dim, pose_dim)
        self.decshape = nn.Linear(hidden_dim, 11)
        self.deccam = nn.Linear(hidden_dim, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1))


    def forward(self, hpose, hshape, hcam, bbox_info,depth_feats=None):
        BN = hpose.shape[0]
        hpose = torch.cat([hpose, bbox_info.unsqueeze(-1).repeat(1,1,22)], 1)
        hshape = torch.cat([hshape, bbox_info.unsqueeze(-1).repeat(1,1,22)], 1)
        hcam = torch.cat([hcam, bbox_info.unsqueeze(-1).repeat(1,1,22)], 1)
        if depth_feats is not None:
            depth_feats = self.avgpool(depth_feats)
            hpose = torch.cat([hpose, depth_feats.squeeze(-1).repeat(1,1,22)], 1)
            hshape = torch.cat([hshape, depth_feats.squeeze(-1).repeat(1,1,22)], 1)
            hcam = torch.cat([hcam, depth_feats.squeeze(-1).repeat(1,1,22)], 1)

        hpose = hpose.transpose(1, 2)
        hshape = hshape.transpose(1, 2)
        hcam = hcam.transpose(1, 2)
        hshape = hshape.flatten(1)
        hcam = hcam.flatten(1)

        #Iteratively update pose and shape and cam
        pred_pose = self.init_pose.repeat(BN, 1).reshape(BN, 22,-1)
        pred_shape = self.init_shape.repeat(BN, 1)
        pred_cam = self.init_cam.repeat(BN, 1)

        for i in range(1):
            pose_feats_pred = torch.cat([pred_pose, hpose], 2)
            shape_feats_pred = torch.cat([pred_shape, hshape], 1)
            cam_feats_pred = torch.cat([pred_cam, hcam], 1)

            pred_pose = pred_pose + self.decpose(self.p(pose_feats_pred))
            pred_shape = pred_shape + self.decshape(self.s(shape_feats_pred))
            pred_cam = pred_cam + self.deccam(self.c(cam_feats_pred))

        d_pose = pred_pose
        d_shape = pred_shape
        d_cam = pred_cam

        #rotmat
        rotm = rot6d_to_rotmat(d_pose).view(BN, 22,3,3)
        output = {
            'pred_pose': rotm,
            'pred_cam': d_cam,
            'pred_shape': d_shape,
            'pred_pose_6d': d_pose,
        }
        return output
    
    def _make_linear(self, num, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [nn.Linear(plane, hidden_dim), 
                     nn.ReLU(inplace=True)]
            layers.extend(layer)  

            plane = hidden_dim

        return nn.Sequential(*layers)
    
    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [MultiLinear(n_head, plane, hidden_dim), 
                     nn.ReLU(inplace=True)]
            layers.extend(layer)
            
            plane = hidden_dim

        return nn.Sequential(*layers)