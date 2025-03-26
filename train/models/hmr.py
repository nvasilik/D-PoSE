import torch
import torch.nn as nn
import numpy as np
from .head.refit_regressor import Regressor
from .head.smplx_cam_head import SMPLXCamHead

from .head.keypoint_attention import KeypointAttention
from ..core.config import PRETRAINED_CKPT_FOLDER
from .head.unet_advanced import UNET
from .backbone.hrnet import hrnet_w32, hrnet_w48
from ..utils.one_euro_filter import OneEuroFilter
class HMR(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            img_res=224,
            focal_length=5000,
            pretrained_ckpt=None,
            hparams=None
    ):
        super(HMR, self).__init__()
        self.hparams = hparams

        # Initialize backbone
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            pretrained_ckpt = backbone + '-' + pretrained_ckpt
            pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
            self.backbone = eval(backbone)(
                pretrained_ckpt_path=pretrained_ckpt_path,
                downsample=True,
                use_conv=(use_conv == 'conv'),
            )

        if hparams.TRIAL.bedlam_bbox:

            self.head = Regressor()
            self.smpl = SMPLXCamHead(img_res=img_res)
        if hparams.DATASET.USE_DEPTH:
            self.depth_decoder = UNET(depth=True)

        if hparams.DATASET.USE_SEGM:
            self.segmentation_decoder = UNET(depth=False)
            self.attention = KeypointAttention()
            self.avg_pool_cam_shape = nn.AdaptiveAvgPool2d((768,1))
        self.min_cutoff=0.004
        self.beta=0.7
        self.t = 0
        self.one_euro_pose = None
        self.one_euro_shape = None#OneEuroFilter(np.zeros(10), min_cutoff=min_cutoff, beta=beta)
        self.one_euro_cam = None#OneEuroFilter(np.zeros(3), min_cutoff=min_cutoff, beta=beta)
        self.use_one_euro = True

    def forward(
            self,
            images,
            bbox_scale=None,
            bbox_center=None,
            img_w=None,
            img_h=None,
            fl=None
    ):
        batch_size = images.shape[0]

        if fl is not None:
            # GT focal length
            focal_length = fl
        else:
            # Estimate focal length
            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            focal_length = focal_length.repeat(2).view(batch_size, 2)

        # Initialze cam intrinsic matrix
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0] = focal_length[:, 0]
        cam_intrinsics[:, 1, 1] = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w/2.
        cam_intrinsics[:, 1, 2] = img_h/2.

        if self.hparams.TRIAL.bedlam_bbox:
            # Taken from CLIFF repository
            cx, cy = bbox_center[:, 0], bbox_center[:, 1]
            b = bbox_scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                    dim=-1)
            bbox_info = bbox_info.cuda().float()
            bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)   # [-1, 1]
            bbox_info[:, 2] = bbox_info[:, 2] / cam_intrinsics[:, 0, 0]  # [-1, 1]
            bbox_info = bbox_info.cuda().float()

            if self.hparams.DATASET.USE_SEGM:
                features,upsampled_feature= self.backbone(images)
                segmentation,_ = self.segmentation_decoder(features)
                cam_shape_feat = upsampled_feature
                if not self.hparams.DATASET.USE_DEPTH:
                    attention_pose = self.attention(upsampled_feature,segmentation[:,1:,:,:],None)
                    attention_cam_shape = self.attention(cam_shape_feat,segmentation[:,1:,:,:],None)
                    hmr_output = self.head(attention_pose,attention_cam_shape,attention_cam_shape,bbox_info)
                else:
                    depth,depth_feats = self.depth_decoder(features)
                    orig_depth = depth.clone()
                    depth = depth.repeat(1,segmentation.shape[1],1,1)
                    if not self.hparams.DATASET.USE_DEPTH_CONC:
                        attention_pose = self.attention(upsampled_feature,segmentation[:,1:,:,:],depth[:,1:,:,:])
                        attention_cam_shape = self.attention(cam_shape_feat,segmentation[:,1:,:,:],depth[:,1:,:,:])
                        hmr_output = self.head(attention_pose,attention_cam_shape,attention_cam_shape,bbox_info,depth_feats)
                    else:
                        upsampled_feature = torch.cat([upsampled_feature,depth_feats],dim=1)
                        attention_pose = self.attention(upsampled_feature,segmentation[:,1:,:,:],depth[:,1:,:,:])
                        cam_shape_feat = torch.cat([cam_shape_feat,depth_feats],dim=1)
                        attention_cam_shape = self.attention(cam_shape_feat,segmentation[:,1:,:,:],depth[:,1:,:,:])
                        hmr_output = self.head(attention_pose,attention_cam_shape,attention_cam_shape,bbox_info,None)
            else:
                features,_= self.backbone(images)
                hmr_output = self.head(features, bbox_info=bbox_info,depth_feats=None)

        if self.hparams.TRIAL.bedlam_bbox:
            # Assuming prediction are in camera coordinate
            #if first time, initialize one euro filter
            if self.one_euro_cam is None and self.use_one_euro:
                self.one_euro_cam = OneEuroFilter(np.zeros_like(hmr_output['pred_cam'][0].cpu()), hmr_output['pred_cam'][0].cpu().numpy(), min_cutoff=self.min_cutoff, beta=self.beta)
                self.one_euro_pose = OneEuroFilter(np.zeros_like(hmr_output['pred_pose'][0].cpu()), hmr_output['pred_pose'][0].cpu().numpy(), min_cutoff=self.min_cutoff, beta=self.beta)
                self.one_euro_shape = OneEuroFilter(np.zeros_like(hmr_output['pred_shape'][0].cpu()), hmr_output['pred_shape'][0].cpu().numpy(), min_cutoff=self.min_cutoff, beta=self.beta)
            #filter the output
            if self.use_one_euro:
                self.t+=1
                t_pose = np.ones_like(hmr_output['pred_pose'][0].cpu()) * self.t
                t_shape = np.ones_like(hmr_output['pred_shape'][0].cpu()) * self.t
                t_cam = np.ones_like(hmr_output['pred_cam'][0].cpu()) * self.t
                #import ipdb; ipdb.set_trace()
                hmr_output['pred_cam'] = torch.tensor(self.one_euro_cam(t_cam,hmr_output['pred_cam'].cpu().numpy())).cuda()
                hmr_output['pred_pose'] = torch.tensor(self.one_euro_pose(t_pose,hmr_output['pred_pose'].cpu().numpy())).cuda()
                hmr_output['pred_shape'] = torch.tensor(self.one_euro_shape(t_shape,hmr_output['pred_shape'].cpu().numpy())).cuda()
            smpl_output = self.smpl(
                rotmat=hmr_output['pred_pose'],
                shape=hmr_output['pred_shape'],
                cam=hmr_output['pred_cam'],
                cam_intrinsics=cam_intrinsics,
                bbox_scale=bbox_scale,
                bbox_center=bbox_center,
                img_w=img_w,
                img_h=img_h,
                normalize_joints2d=False,
            )
        smpl_output.update(hmr_output)
        if self.hparams.DATASET.USE_DEPTH or self.hparams.DATASET.USE_SEGM:
            if self.hparams.DATASET.USE_DEPTH:
                return smpl_output,orig_depth,None,None,segmentation
            else:
                return smpl_output,None,None,None,segmentation
        return smpl_output
