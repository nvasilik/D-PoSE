import torch
import torch.nn as nn
from ..utils.geometry import batch_rodrigues, rotmat_to_rot6d,estimate_translation
from ..utils.eval_utils import compute_similarity_transform_batch
from ..core.constants import NUM_JOINTS_SMPLX, NUM_JOINTS_SMPL
import torch.nn.functional as F
import neural_renderer as nr
import numpy as np
import json
from trimesh.visual import color
from ..utils.image_utils import crop
import torch.nn.functional as F
import time
from torchmetrics.image import StructuralSimilarityIndexMeasure
def create_affine_transform(centers, scales, device, target_size):
    """
    Create an affine transformation that aligns with the cropping and scaling.
    
    Parameters:
        centers (Tensor): The centers around which to crop, shape (B, 2).
        scales (Tensor): The scaling factors for each crop, shape (B, 1).
        device (torch.device): The device on which tensors are allocated.
        target_size (List[int]): The target size of the crop [H, W].
        
    Returns:
        Tensor: The affine transformation matrices, shape (B, 2, 3).
    """
    B = centers.size(0)
    
    # Calculate the scale relative to the target size
    scale_mat = torch.zeros((B, 2, 2), device=device)
    scale = scales / max(target_size) * 2.0  # Scale factor normalized for grid_sample usage
    scale_mat[:, 0, 0] = scale.squeeze()
    scale_mat[:, 1, 1] = scale.squeeze()

    # Calculate translations
    trans_mat = torch.zeros((B, 2, 3), device=device)
    trans_mat[:, :, 2] = centers * 2.0 / torch.tensor(target_size, device=device) - 1.0  # Normalize center coordinates
    trans_mat[:, :, 2] -= scale  # Adjust translation based on scale

    # Combine scale and translation
    affine_mat = scale_mat @ trans_mat[:, :, :2]
    affine_mat = torch.cat((affine_mat, trans_mat[:, :, 2:]), dim=2)
    
    return affine_mat

def batched_crop_and_resize(images, centers, scales, target_size):
    """
    Batch crop and resize images using grid sampling.
    
    Parameters:
        images (Tensor): The input images, shape (B, C, H, W).
        centers (Tensor): The centers for cropping, shape (B, 2).
        scales (Tensor): The scales for cropping, shape (B, 1).
        target_size (List[int]): The target size [H, W] for the output images.
        
    Returns:
        Tensor: The cropped and resized images, shape (B, C, target_size[0], target_size[1]).
    """
    device = images.device
    B, C, H, W = images.shape
    affine_mat = create_affine_transform(centers, scales, device, target_size)
    
    # Create a grid for the target size
    grid = F.affine_grid(affine_mat, [B, C, *target_size], align_corners=False)
    
    # Sample the input images using the grid
    cropped_and_resized = F.grid_sample(images, grid, align_corners=False)
    
    return cropped_and_resized
@torch.no_grad()
def get_body_part_texture(faces, n_vertices,part_segm):
    faces = faces.detach().cpu().numpy()
    nparts = 22
    grayscale_values = (torch.arange(int(nparts)) / float(nparts) * 255.) + 1
    batch_size = faces.shape[0]
    vertex_colors = torch.ones((n_vertices, 3))  # Include alpha for completeness
    for part_idx, (_, vertices) in enumerate((part_segm.items())):
        # Ensure each vertex assigned to this part gets the correct grayscale value
        # and full opacity in the alpha channel
        vertex_colors[vertices, :3] = grayscale_values[part_idx]  # Apply grayscale value to RGB
        #vertex_colors[vertices, 3] = 255  # Full opacity
  

    vertex_colors = color.to_rgba(vertex_colors)
    #vertex_colors = torch.from_numpy(vertex_colors).float().cuda()
    face_colors = vertex_colors[faces].min(axis=2)

    texture = np.zeros((batch_size, faces.shape[1], 1, 1, 1, 3), dtype=np.float32)
    texture[:, :, 0, 0, 0, :] = face_colors[:,:, :3] #/ nparts
    texture = torch.from_numpy(texture).float().cuda()
    return texture

@torch.no_grad()
def get_default_camera(focal_length, img_size):
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1
    K[0, 2] = img_size / 2.
    K[1, 2] = img_size / 2.
    K = K[None, :, :]
    R = torch.eye(3)[None, :, :]
    return K, R

def gaussian_window(size, sigma):
    """
    Generate a Gaussian window used for the SSIM calculation.
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

def ssim(img1, img2, window_size=8, window_sigma=1.5, size_average=True, val_range=1.0):
    """
    Compute the SSIM between two images.
    """
    L = val_range  # The dynamic range of the pixel-values (1 for [0,1])
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    window = gaussian_window(window_size, window_sigma).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def ssim_loss(img1, img2):
    """
    Compute the SSIM loss between two images.
    """
    return 1 - ssim(img1, img2)
# Example usage
# Assuming `image1` and `image2` are your input tensors with shape [batch_size, 1, height, width] and values in [0, 1].
# loss = ssim_loss(image1, image2)

@torch.no_grad()
def generate_part_labels(vertices, faces_colors,faces, neural_renderer, K,R, part_bins,cam_t=None):
    batch_size = vertices.shape[0]
    #body_part_texture, _ = part_segm_to_vertex_colors(part_segm, vertices.shape[1])
    #K batched
    '''
    K_ = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
    for i in range(batch_size):
        K_[i, 0, 0] = focal_length[i, 0]
        K_[i, 1, 1] = focal_length[i, 1]
        K_[i, 0, 2] = img_w[i]/2.
        K_[i, 1, 2] = img_h[i]/2.
    #vertex_colors = vertex_colors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    R_ = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
    '''
    K_ = K.repeat(batch_size, 1, 1).cuda().float()
    R_ = R.repeat(batch_size, 1, 1).cuda().float()

    faces_colors = faces_colors.cuda()
    #t = translation[:, :3].unsqueeze(1).cuda().float()
    #t = torch.zeros(1, 3).cuda().float()
    #t_
    body_parts, depth, mask = neural_renderer(
        vertices=vertices,
        faces=faces,
        K=K_,
        R=R_,
        t=cam_t,
        textures=faces_colors,
    )

    #plt.imshow(body_parts[0].detach().cpu().numpy().transpose(1, 2, 0)*255)
    #plt.savefig('body_parts.png')
    #render_rgb = body_parts.clone()

    body_parts = body_parts.permute(0, 2, 3, 1)
    #body_parts *= 255. # multiply it with 255 to make labels distant
    body_parts, _ = body_parts.max(-1) # reduce to single channel
    body_parts = body_parts.long()
    body_parts = torch.bucketize(body_parts, part_bins, right=True) # np.digitize(body_parts, bins, right=True)

    # add 1 to make background label 0
    body_parts = (body_parts.long()+1) * mask.long()
    '''
    import matplotlib.pyplot as plt
    plt.imshow(body_parts[0].long().detach().cpu().numpy())
    plt.savefig('body_parts.png')
    import ipdb;ipdb.set_trace()
    '''

    #crop it to the image size
    #body_parts = body_parts[:, :img_h, :img_w]
    #mask = mask[:, :img_h, :img_w]
    '''
    new_body_parts = []
    new_mask = []
    for i in range(batch_size):
        new_body_parts.append( body_parts[i, :img_h[i], :img_w[i]])
        new_mask.append(mask[i, :img_h[i], :img_w[i]])
    body_parts = torch.stack(new_body_parts)
    mask = torch.stack(new_mask) 
    '''   
    #body_parts = body_parts * mask.detach()
    #body_parts = body_parts*mask
    #cropped= crop(body_parts.detach().cpu().numpy(), center.detach().cpu().numpy(), scale.detach().cpu().numpy(), [224, 224])
    '''
    start = time.time()
    cropped = np.zeros((batch_size, 224, 224))
    for i in range(batch_size):
        cropped[i] = crop(body_parts[i].detach().cpu().numpy(), center[i].detach().cpu().numpy(), scale[i].detach().cpu().numpy(), [224, 224])
    #body_parts = batched_crop_and_resize(body_parts.unsqueeze(1).float(), center, scale, [224, 224])
    cropped = torch.from_numpy(cropped).long()
    end = time.time()
    print('Time taken for cropping:', end-start)
    '''
    #cropped = torch.zeros((batch_size, 224, 224))
    #One hot encoding
    one_hot = F.one_hot(body_parts, 23).permute(0, 3, 1, 2)


    return one_hot.float()
class HMRLoss(nn.Module):
    def __init__(
            self,
            hparams=None,
    ):
        super(HMRLoss, self).__init__()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mse_noreduce = nn.MSELoss(reduction='none')
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l1_noreduce = nn.L1Loss(reduction='none')
        self.criterion_cross_entropy =nn.CrossEntropyLoss()
        self.hparams = hparams

        #self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.loss_weight = self.hparams.MODEL.LOSS_WEIGHT
        self.shape_loss_weight = self.hparams.MODEL.SHAPE_LOSS_WEIGHT
        self.pose_loss_weight = self.hparams.MODEL.POSE_LOSS_WEIGHT
        self.joint_loss_weight = self.hparams.MODEL.JOINT_LOSS_WEIGHT
        self.keypoint_loss_weight_2d = self.hparams.MODEL.KEYPOINT_LOSS_WEIGHT
        self.beta_loss_weight = self.hparams.MODEL.BETA_LOSS_WEIGHT
        if self.hparams.TRIAL.version == 'real': # Using SMPL
            self.num_joints = 49
        else:
            self.num_joints = 24
        if self.hparams.DATASET.SUPERVISE_SEGM:
            with torch.no_grad():
                self.neural_renderer =  nr.Renderer(
                    dist_coeffs=None,
                    #orig_size=(1280, 720),
                    orig_size=224,
                    image_size=224,
                    light_intensity_ambient=1,
                    light_intensity_directional=0,
                    light_color_ambient=(1, 1, 1),
                    light_color_directional=(0, 0, 0),
                    anti_aliasing=False,
                )            
                with open('part-segm.json') as f:
                    self.part_segm_json = json.load(f)
                #self.vertex_colors, self.grayscale_values = part_segm_to_vertex_colors(self.part_segm_json, 10475)
                #import ipdb;ipdb.set_trace()
                self.num_verts = 10475
                n_parts = 22
                self.part_bins = (torch.arange(int(n_parts)) / float(n_parts) * 255.) + 1
                #self.part_bins = torch.linspace(0, 255, 23)+1
                self.part_bins = self.part_bins.cuda()
                self.K,self.R = get_default_camera(5000.0, 224)
                self.faces_colors = None
                #self.part_bins = torch.linspace(0, 0.88, 23)
                self.first_forward = True

    def forward(self, pred, gt):
        if self.hparams.TRIAL.criterion == 'mse':
            self.criterion = self.criterion_mse
            self.criterion_noreduce = self.criterion_mse_noreduce
        elif self.hparams.TRIAL.criterion == 'l1':
            self.criterion = self.criterion_l1
            self.criterion_noreduce = self.criterion_l1_noreduce
        if self.hparams.DATASET.SUPERVISE_SEGM and self.first_forward:
            self.first_forward = False
            faces = gt['faces']
            with torch.no_grad():
                self.faces_colors = get_body_part_texture(faces, self.num_verts ,self.part_segm_json)
        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['joints3d'][:, :self.num_joints]
        pred_keypoints_2d = pred['joints2d'][:, :self.num_joints]
        pred_vertices = pred['vertices']
        gt_betas = gt['betas']
        gt_joints = gt['joints3d'][:, :self.num_joints]
        gt_vertices = gt['vertices']
        gt_pose = gt['pose']
        if self.hparams.DATASET.SUPERVISE_DEPTH:
            pred_depth = pred['depth']
            gt_depth = gt['depth']
            has_depth = gt['has_depth']
            l1_loss = depth_loss(pred_depth[has_depth], gt_depth[has_depth], self.criterion)
            ssim_loss_ = 0.5*ssim_loss(pred_depth[has_depth], gt_depth[has_depth])
        if self.hparams.DATASET.DEPTH_ONLY:
            loss_dict = {}
            loss_dict['loss/loss_depth_l1'] = l1_loss
            loss_dict['loss/loss_ssim'] = ssim_loss_
            loss = (l1_loss)*self.loss_weight
            loss_dict['loss/loss'] = loss
            return loss, loss_dict
        if self.hparams.DATASET.SUPERVISE_SEGM:
            part_verts = gt['vertices_obj']
            focal_length = gt['focal_length']
            scale = gt['scale']
            center = gt['center']
            orig_img_size = gt['orig_shape']
            pred_segms = pred['part_segms']
            #camera_translation = gt['translation']
            '''
            gt_joints = gt['joints3d'][:, :self.num_joints]
            gt_keypoints_2d = gt['keypoints']
            gt_keypoints_2d[:, :, :-1] = \
            0.5 * self.hparams.DATASET.IMG_RES * (gt_keypoints_2d[:, :, :-1] + 1)

            #import ipdb;ipdb.set_trace()
            gt_cam_t = estimate_translation(
                gt_joints.detach().cpu(),
                gt_keypoints_2d.detach().cpu(),
                focal_length=5000.0,
                img_size=224,
                use_all_joints=True,
            )
            '''
            gt_cam_t = gt['parts_cam_t']
            with torch.no_grad():
                faces = gt['faces']
                #gt_segms = generate_part_labels(part_verts.detach(),faces_colors.detach(),faces.detach(), self.neural_renderer, focal_length.detach(), self.part_bins.detach(),orig_img_size.detach(),scale.detach(),center.detach(),cam_t=gt_cam_t).cuda()
                gt_segms = generate_part_labels(part_verts.detach(),self.faces_colors.detach(),faces.detach(), self.neural_renderer, self.K,self.R, self.part_bins,cam_t=gt_cam_t).cuda()
                has_segms = gt_segms.sum(dim=(1, 2, 3)) > 0
            cross_entropy_loss = self.criterion_cross_entropy(pred_segms[has_segms], gt_segms[has_segms])            

        if self.hparams.TRIAL.bedlam_bbox:
            # Use full image keypoints
            pred_keypoints_2d[:, :, :2] = 2 * (pred_keypoints_2d[:, :, :2] / img_size) - 1
            gt_keypoints_2d = gt['keypoints_orig']
            gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] / img_size) - 1
        else:
            # Use crop keypoints
            gt_keypoints_2d = gt['keypoints']

        loss_keypoints = projected_keypoint_loss(
            pred_keypoints_2d,
            gt_keypoints_2d,
            criterion=self.criterion_noreduce,
        )

        if self.hparams.TRIAL.bedlam_bbox:
            loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
            loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
            loss_keypoints = loss_keypoints.mean()
        else:
            loss_keypoints = loss_keypoints.mean()

        # Compute 2D reprojection loss for the keypoints
        if self.hparams.DATASET.proj_verts:
            pred_proj_verts = pred['pred_proj_verts']
            gt_proj_verts = gt['proj_verts_orig']
            gt_proj_verts[:, :, :2] = 2 * (gt_proj_verts[:, :, :2] / img_size) - 1
            pred_proj_verts[:, :, :2] = 2 * (pred_proj_verts[:, :, :2] / img_size) - 1

            loss_projverts = projected_verts_loss(
                pred_proj_verts,
                gt_proj_verts,
                criterion=self.criterion_noreduce,
            )

            if self.hparams.TRIAL.bedlam_bbox:
                loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
                loss_projverts = loss_projverts * loss_keypoints_scale.unsqueeze(1)
                loss_projverts = loss_projverts.mean()
            else:
                loss_projverts = loss_projverts.mean()
  
        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            criterion=self.criterion,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            criterion=self.criterion,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            criterion=self.criterion_l1,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight_2d
        loss_keypoints_3d *= self.joint_loss_weight

        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        pred_cam_clipped = torch.clamp(pred_cam[:, 0], min=-0.5, max=0.5)
        loss_cam = ((torch.exp(-pred_cam_clipped * 10)) ** 2).mean()

        if self.hparams.TRIAL.losses_abl == 'param':
            loss_dict = {
                        'loss/loss_keypoints': loss_keypoints,
                        'loss/loss_regr_pose': loss_regr_pose,
                        'loss/loss_regr_betas': loss_regr_betas,
                        'loss/loss_cam': loss_cam,
                    }
        elif self.hparams.TRIAL.losses_abl == 'param_keypoints':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.TRIAL.losses_abl == 'keypoints':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.TRIAL.losses_abl == 'param_verts':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                # 'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.TRIAL.losses_abl == 'verts':
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                # 'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }
        elif self.hparams.DATASET.proj_verts:
            loss_projverts *= self.keypoint_loss_weight_2d

            loss_dict = {
                'loss/loss_projverts': loss_projverts,
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }

        else: # param+keypoints+verts
            loss_dict = {
                'loss/loss_keypoints': loss_keypoints,
                'loss/loss_keypoints_3d': loss_keypoints_3d,
                'loss/loss_regr_pose': loss_regr_pose,
                'loss/loss_regr_betas': loss_regr_betas,
                'loss/loss_shape': loss_shape,
                'loss/loss_cam': loss_cam,
            }

        if self.hparams.DATASET.SUPERVISE_SEGM:
            loss_body = sum(loss for loss in loss_dict.values())
            loss_dict['loss/loss_cross_entropy'] = cross_entropy_loss
            #import ipdb;ipdb.set_trace()
            if self.hparams.DATASET.SUPERVISE_DEPTH:
                depth_loss_= l1_loss+0.2*ssim_loss_
                loss_dict['loss/loss_depth_l1'] = depth_loss_
                loss_dict['loss/loss_depth_ssim'] = ssim_loss_
                loss = (loss_body*10 + cross_entropy_loss*0.1 + depth_loss_*0.1)*self.loss_weight
            else:
                loss = (loss_body*10 + cross_entropy_loss*0.1)*self.loss_weight
        elif self.hparams.DATASET.SUPERVISE_DEPTH:
            loss_body = sum(loss for loss in loss_dict.values())
            depth_loss_= l1_loss+0.2*ssim_loss_
            loss_dict['loss/loss_depth_l1'] = depth_loss_
            loss_dict['loss/loss_depth_ssim'] = ssim_loss_
            loss = (loss_body*10 + depth_loss_*0.1)*self.loss_weight
        else:
            loss = sum(loss for loss in loss_dict.values())
            loss *= self.loss_weight
        loss_dict['loss/loss'] = loss
        return loss, loss_dict


def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):

    conf = gt_keypoints_2d[:, :, -1]
    conf[conf == -2] = 0
    conf = conf.unsqueeze(-1)
    loss = conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
    return loss

def projected_verts_loss(
        pred_proj_verts,
        gt_proj_verts,
        criterion,
):
    loss = criterion(pred_proj_verts, gt_proj_verts[:, :, :-1])
    return loss


def keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):

    loss = criterion(pred_keypoints_2d, gt_keypoints_2d)
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        criterion,
):
    gt_keypoints_3d = gt_keypoints_3d.clone()
    pred_keypoints_3d = pred_keypoints_3d
    if len(gt_keypoints_3d) > 0:
        return (criterion(pred_keypoints_3d, gt_keypoints_3d))
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


def reconstruction_error(pred_keypoints_3d, gt_keypoints_3d, criterion):
    pred_keypoints_3d = pred_keypoints_3d.detach().cpu().numpy()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].detach().cpu().numpy()

    pred_keypoints_3d_hat = compute_similarity_transform_batch(pred_keypoints_3d, gt_keypoints_3d)
    return criterion(torch.tensor(pred_keypoints_3d_hat), torch.tensor(gt_keypoints_3d)).mean()


def shape_loss(
        pred_vertices,
        gt_vertices,
        criterion,
):
    pred_vertices_with_shape = pred_vertices
    gt_vertices_with_shape = gt_vertices

    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


def smpl_losses(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        criterion,
):
    pred_rotmat_valid = pred_rotmat[:,1:]
    batch_size = pred_rotmat_valid.shape[0]
    gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3)[:, 1:]
    pred_betas_valid = pred_betas
    gt_betas_valid = gt_betas

    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = (criterion(pred_rotmat_valid, gt_rotmat_valid))
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas

def depth_loss(
        pred_depth,
        gt_depth,
        criterion,
):
    loss = criterion(pred_depth, gt_depth)
    return loss
