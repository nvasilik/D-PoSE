import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch
import pickle
import joblib
import json
import numpy as np
from loguru import logger
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from skimage.transform import resize
from ..core import constants, config
from ..core.constants import NUM_JOINTS_SMPLX
from ..core.config import DATASET_FILES, DATASET_FOLDERS, DEPTH_FOLDERS, MASK_FOLDERS
from ..utils.image_utils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, random_crop, read_img,denormalize_images
from ..utils.geometry import estimate_translation
from ..utils.depth_utils import read_depth_percentile_norm,read_depth_exr,read_body_clothing_segm_mask,read_human_background_depth,read_human_only_depth,part_segm_to_vertex_colors,SegmRenderer,read_human_parts_depth
from smplx import SMPL, SMPLX
from albumentations.core.transforms_interface import ImageOnlyTransform

class CustomRandomOcclusion(ImageOnlyTransform):
    def __init__(self, occlusion_size_min=(10, 10), occlusion_size_max=(40, 40), p=0.2):
        super().__init__(p=p)
        self.occlusion_size_min = occlusion_size_min
        self.occlusion_size_max = occlusion_size_max

    def apply(self, img, **params):
        height, width = img.shape[:2]
        occlusion_size = (
            np.random.randint(self.occlusion_size_min[0], self.occlusion_size_max[0]),
            np.random.randint(self.occlusion_size_min[1], self.occlusion_size_max[1])
        )
        top = np.random.randint(0, height - occlusion_size[0])
        left = np.random.randint(0, width - occlusion_size[1])

        # Calculate min and max pixel values in the image
        img_min = img.min()
        img_max = img.max()

        # Ensure img_min and img_max are suitable for the data type of the image
        if img.dtype == np.uint8:
            # For uint8 images, values are already in the [0, 255] range
            noise = np.random.randint(img_min, img_max + 1, 
                                      (occlusion_size[0], occlusion_size[1], img.shape[2]), 
                                      dtype=np.uint8)
        else:
            # For floating-point images, scale random values to [img_min, img_max]
            noise = np.random.uniform(img_min, img_max, 
                                      (occlusion_size[0], occlusion_size[1], img.shape[2])).astype(img.dtype)

        img[top:top + occlusion_size[0], left:left + occlusion_size[1], :] = noise
        return img

class DatasetHMR(Dataset):

    def __init__(self, options, dataset, use_augmentation=True, is_train=True):
        super(DatasetHMR, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                       std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset],
                            allow_pickle=True)
        # Bounding boxes are assumed to be in the center and scale format
        if 'emdb' in dataset:
            self.scale = self.data['scale']
            self.center = self.data['center']
            self.imgname = self.data['imgname']
            self.pose_cam = self.data['body_pose']
            self.betas = self.data['betas']
            self.gender = np.array([0 if str(g) == 'male'
                                        else 1 for g in self.data['gender']]).astype(np.int32)
        else:
            self.scale = self.data['scale']
            self.center = self.data['center']
            self.imgname = self.data['imgname']


        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        if use_augmentation:
            #self.occluders = joblib.load(PASCAL_OCCLUDERS)
            self.occlusion_prob = 0.5 
        with open('part-segm.json') as f:
            self.smplx_part_segm = json.load(f)
        

        #Overwrite use_augmentation to False if USE_DEPTH is True
        if self.is_train and self.options.USE_DEPTH and self.options.SUPERVISE_DEPTH:
            self.depth_dir = DEPTH_FOLDERS[dataset]
            self.mask_dir = MASK_FOLDERS[dataset]
            self.depthname = [s.replace('.png', '_depth.exr') for s in self.imgname]

            self.maskname = [s.replace('.png', '') for s in self.imgname]


        if self.is_train:
            if '3dpw-train-smplx' in self.dataset:
                self.pose_cam = self.data['smplx_pose'][:, :NUM_JOINTS_SMPLX*3].astype(np.float)
                self.betas = self.data['smplx_shape'][:, :11].astype(np.float)
            else:
                self.pose_cam = self.data['pose_cam'][:, :NUM_JOINTS_SMPLX*3].astype(np.float)
                self.betas = self.data['shape'].astype(np.float)
            # For AGORA and 3DPW num betas are 10
            if self.betas.shape[-1] == 10:
                self.betas = np.hstack((self.betas, np.zeros((self.betas.shape[0], 1))))

            if 'cam_int' in self.data:
                self.cam_int = self.data['cam_int']
            else:
                self.cam_int = np.zeros((self.imgname.shape[0], 3, 3))
            if 'cam_ext' in self.data:
                self.cam_ext = self.data['cam_ext']
            else:
                self.cam_ext = np.zeros((self.imgname.shape[0], 4, 4))
            if 'trans_cam' in self.data:
                self.trans_cam = self.data['trans_cam']
        elif 'orbit-stadium-bmi' in self.dataset:
            self.pose_cam = self.data['pose_cam'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            self.joints = self.data['gtkps']
            self.joints = self.joints[:, :24]
            #import ipdb; ipdb.set_trace()
        elif not 'emdb' in self.dataset:
            if 'h36m' in self.dataset: # H36m doesn't have pose and shape param only 3d joints
                self.joints = self.data['S']
                self.pose_cam = np.zeros((self.imgname.shape[0], 66))
                self.betas = np.zeros((self.imgname.shape[0], 11))
            else:
                self.pose_cam = self.data['pose_cam'].astype(np.float)
                self.betas = self.data['shape'].astype(np.float)
        if self.is_train:
            if '3dpw-train-smplx' in self.dataset: # Only for 3dpw training
                self.joint_map = constants.joint_mapping(constants.COCO_18, constants.SMPL_24)
                self.keypoints = np.zeros((len(self.imgname), 24, 3))
                self.keypoints = self.data['gtkps'][:, self.joint_map]
                self.keypoints[:, self.joint_map == -1] = -2
            else:
                full_joints = self.data['gtkps']
                self.keypoints = full_joints[:, :24]
        else:
            self.keypoints = np.zeros((len(self.imgname), 24, 3))

        if 'proj_verts' in self.data:
            self.proj_verts = self.data['proj_verts']
        else:
            self.proj_verts = np.zeros((len(self.imgname), 437, 3))
        if not 'emdb' in self.dataset:
            try:
                gender = self.data['gender']
                self.gender = np.array([0 if str(g) == 'm'
                                        else 1 for g in gender]).astype(np.int32)
            except KeyError:
                self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        # evaluation variables
        if True:
            if 'width' in self.data: # For closeup image stored in rotated format
                self.width = self.data['width']
            self.joint_mapper_h36m = constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(
                               config.JOINT_REGRESSOR_H36M)).float()
            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                                  gender='male',
                                  create_transl=False)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                                    gender='female',
                                    create_transl=False)
            self.smplx_male = SMPLX(config.SMPLX_MODEL_DIR,
                                    gender='male')
            self.smplx_female = SMPLX(config.SMPLX_MODEL_DIR,
                                      gender='female')
            self.smplx_male_bedlam = SMPLX(config.SMPLX_MODEL_DIR,
                                      gender='male',num_betas=11)
            self.smplx_female_bedlam = SMPLX(config.SMPLX_MODEL_DIR,
                                      gender='female',num_betas=11)
            self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, 'rb'))
            self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None],
                                           dtype=torch.float32)
        if self.is_train and 'agora' not in self.dataset and '3dpw' not in self.dataset: # first 80% is training set 20% is validation
            self.length = int(self.scale.shape[0] * self.options.CROP_PERCENT)
        else:
            self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def scale_aug(self):
        sc = 1            # scaling
        if self.is_train:
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.SCALE_FACTOR,
                    max(1-self.options.SCALE_FACTOR,
                    np.random.randn()*self.options.SCALE_FACTOR+1))
        return sc
    def rgb_processing(self, rgb_img_full, center, scale, img_res, kp2d=None):

        if self.is_train and self.options.ALB and self.use_augmentation:
            #if np.random.uniform() <= self.occlusion_prob:
                #rgb_img = occlude_with_pascal_objects(rgb_img_full, self.occluders)
            aug_comp = [CustomRandomOcclusion(occlusion_size_min=(50, 50), occlusion_size_max=(100, 100), p=1.0),
                        A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                        A.ImageCompression(20, 100, p=0.1),
                        A.RandomRain(blur_value=4, p=0.1),
                        A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                        A.Blur(blur_limit=(3, 10), p=0.1),
                        A.RandomSnow(brightness_coeff=1.5,
                        snow_point_lower=0.2, snow_point_upper=0.4)]
            aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
                       A.RandomBrightnessContrast(p=0.2),
                       A.MultiplicativeNoise(multiplier=[0.5, 1.5],
                       elementwise=True, per_channel=True, p=0.2),
                       A.HueSaturationValue(hue_shift_limit=20,
                       sat_shift_limit=30, val_shift_limit=20,
                       always_apply=False, p=0.2),
                       A.Posterize(p=0.1),
                       A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                       A.Equalize(mode='cv', p=0.1)]
            albumentation_aug = A.Compose([A.OneOf(aug_comp,
                                           p=self.options.ALB_PROB),
                                           A.OneOf(aug_mod,
                                           p=self.options.ALB_PROB)])          
            rgb_img_full = albumentation_aug(image=rgb_img_full)['image']

        rgb_img = crop(rgb_img_full, center, scale, [img_res, img_res])

        rgb_img = np.transpose(rgb_img.astype('float32'),
                               (2, 0, 1))/255.0

        return rgb_img
    



    
    def rgb_depth_processing(self, rgb_img_full,depth_img_full,depth_img_full_bg, center, scale, img_res, kp2d=None):
        if self.is_train and self.options.ALB and self.use_augmentation:
            #aug_occ = [CustomRandomOcclusion(occlusion_size_min=(30, 30), occlusion_size_max=(100, 100), p=self.options.OCCLUSION_PROB)]
            
            aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                        A.ImageCompression(20, 100, p=0.1),
                        A.RandomRain(blur_value=4, p=0.1),
                        A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                        A.Blur(blur_limit=(3, 10), p=0.1),
                        A.RandomSnow(brightness_coeff=1.5,
                        snow_point_lower=0.2, snow_point_upper=0.4)]
            aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
                       A.RandomBrightnessContrast(p=0.2),
                       A.MultiplicativeNoise(multiplier=[0.5, 1.5],
                       elementwise=True, per_channel=True, p=0.2),
                       A.HueSaturationValue(hue_shift_limit=20,
                       sat_shift_limit=30, val_shift_limit=20,
                       always_apply=False, p=0.2),
                       A.Posterize(p=0.1),
                       A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                       A.Equalize(mode='cv', p=0.1)]
            albumentation_aug = A.Compose([A.OneOf(aug_comp,
                                            p=self.options.ALB_PROB),
                                        A.OneOf(aug_mod,
                                        p=self.options.ALB_PROB)])     
            rgb_img_full = albumentation_aug(image=rgb_img_full)['image']   

        rgb_img = crop(rgb_img_full, center, scale, [img_res, img_res])
        if self.is_train and self.use_augmentation:
            #if np.random.uniform() <= self.occlusion_prob:
            #    rgb_img = occlude_with_pascal_objects(rgb_img, self.occluders)
            pass

        rgb_img = np.transpose(rgb_img.astype('float32'),
                               (2, 0, 1))/255.0
        depth_img=crop(depth_img_full, center, scale, [img_res, img_res])
        if depth_img.max() == 0:
            depth_img = np.zeros(depth_img.shape)
        elif depth_img.shape[0] != self.options.DEPTH_MAP_SIZE:
            depth_img = resize(depth_img, (self.options.DEPTH_MAP_SIZE,self.options.DEPTH_MAP_SIZE), anti_aliasing=True)
        depth_img = np.nan_to_num(depth_img,nan=0.0).reshape(depth_img.shape[0], depth_img.shape[1], 1)
        depth_imgs = np.transpose(depth_img.astype('float32'),(2, 0, 1))

        return rgb_img,depth_imgs,None
    
    def get_vertices(self,body_pose,betas,cam_trans,index):
        joints = None
        vertices= None
        if not self.is_train:
            if '3dpw' in self.dataset or 'emdb' in self.dataset:
                if self.gender[index] == 1:
                    model = self.smpl_female

                    gt_smpl_out = self.smpl_female(
                                    global_orient=body_pose.unsqueeze(0)[:, :3],
                                    body_pose=body_pose.unsqueeze(0)[:, 3:],
                                    betas=betas.unsqueeze(0))
                    gt_vertices = gt_smpl_out.vertices
                else:
                    model = self.smpl_male

                    gt_smpl_out = self.smpl_male(
                            global_orient=body_pose.unsqueeze(0)[:, :3],
                            body_pose=body_pose.unsqueeze(0)[:, 3:],
                            betas=betas.unsqueeze(0))
                    gt_vertices = gt_smpl_out.vertices

                vertices = gt_vertices[0].float()
                #joints = torch.matmul(model.J_regressor, gt_vertices[0]).unsqueeze(0)
                joints = gt_smpl_out.joints #TODO: check if this is correct
            elif 'rich' in self.dataset:
                if self.gender[index] == 1:
                    model = self.smpl_female
                    gt_smpl_out = self.smplx_female(
                                    global_orient=body_pose.unsqueeze(0)[:, :3],
                                    body_pose=body_pose.unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                                    betas=betas.unsqueeze(0),)
                    gt_vertices = gt_smpl_out.vertices 
                else:
                    model = self.smpl_male
                    gt_smpl_out = self.smplx_male(
                            global_orient=body_pose.unsqueeze(0)[:, :3],
                            body_pose=body_pose.unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                            betas=betas.unsqueeze(0),
                        )
                    gt_vertices = gt_smpl_out.vertices
                gt_vertices = torch.matmul(self.smplx2smpl, gt_vertices)
                
                joints = torch.matmul(model.J_regressor, gt_vertices[0]).unsqueeze(0)
                vertices = gt_vertices[0].float()
            elif 'h36m' in self.dataset:
                joints = self.joints[index]
                vertices = torch.zeros((6890, 3)).float()
            else:
                vertices = torch.zeros((6890, 3)).float()
                joints = self.joints[index]
        elif  'agora' in self.dataset:
            if self.gender[index] == 1:
                gt_smplx_out = self.smplx_female_bedlam(
                                global_orient=body_pose.unsqueeze(0)[:, :3],
                                body_pose=body_pose.unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                                betas=betas.unsqueeze(0))#,
                                #transl=torch.tensor(cam_trans)[0:3].unsqueeze(0))
                gt_vertices = gt_smplx_out.vertices
            else:
                gt_smpl_out = self.smplx_male_bedlam(
                        global_orient=body_pose.unsqueeze(0)[:, :3],
                        body_pose=body_pose.unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                        betas=betas.unsqueeze(0))#,
                       # transl=torch.tensor(cam_trans)[0:3].unsqueeze(0))
                gt_vertices = gt_smplx_out.vertices

            vertices = gt_vertices[0].float()
            joints = gt_smplx_out.joints
        elif  'agora'not in self.dataset and '3dpw' not in self.dataset:
            if self.gender[index] == 1:
                gt_smplx_out = self.smplx_female_bedlam(
                                global_orient=body_pose.unsqueeze(0)[:, :3],
                                body_pose=body_pose.unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                                betas=betas.unsqueeze(0))#,
                                #transl=torch.tensor(cam_trans)[0:3].unsqueeze(0))
                gt_vertices = gt_smplx_out.vertices
            else:
                gt_smpl_out = self.smplx_male_bedlam(
                        global_orient=body_pose.unsqueeze(0)[:, :3],
                        body_pose=body_pose.unsqueeze(0)[:, 3:NUM_JOINTS_SMPLX*3],
                        betas=betas.unsqueeze(0))#,
                        #transl=torch.tensor(cam_trans)[0:3].unsqueeze(0))
                gt_vertices = gt_smplx_out.vertices
            vertices = gt_vertices[0].float()
            joints = gt_smplx_out.joints
        elif '3dpw-train-smplx' in self.dataset:
            if self.gender[index] == 1:
                gt_smpl_out = self.smpl_female(
                                    global_orient=body_pose.unsqueeze(0)[:, :3],
                                    body_pose=body_pose.unsqueeze(0)[:, 3:],
                                    betas=betas.unsqueeze(0))#,
                                    #transl=torch.tensor(cam_trans).unsqueeze(0))
                gt_vertices = gt_smpl_out.vertices
            else:
                gt_smpl_out = self.smpl_male(
                            global_orient=body_pose.unsqueeze(0)[:, :3],
                            body_pose=body_pose.unsqueeze(0)[:, 3:],
                            betas=betas.unsqueeze(0))#,
                            #transl=torch.tensor(cam_trans).unsqueeze(0))
                gt_vertices = gt_smpl_out.vertices

            vertices = gt_vertices[0].float()
            joints = gt_smpl_out.joints
        
        return vertices,joints[0]

        

    def j2d_processing(self, kp, center, scale):
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [self.options.IMG_RES,
                                   self.options.IMG_RES])
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()
        keypoints_orig_copy = self.keypoints[index].copy()
        if self.options.proj_verts:
            proj_verts_orig = self.proj_verts[index].copy()
            item['proj_verts_orig'] = torch.from_numpy(proj_verts_orig).float()
            proj_verts = self.proj_verts[index].copy()
        # Apply scale augmentation
        sc = self.scale_aug()
        # apply crop augmentation
        if self.is_train and self.options.CROP_FACTOR > 0:
            rand_no = np.random.rand()
            if rand_no < self.options.CROP_PROB:
                center, scale = random_crop(center, scale,
                                            crop_scale_factor=1-self.options.CROP_FACTOR,
                                            axis='y')
        #print('____________'*10)
        #print('HERE'*10)
        if not self.is_train:
            cam_trans=None
            vertices,joints = self.get_vertices(torch.tensor(self.pose_cam[index].copy()).float(),torch.tensor(self.betas[index].copy()).float(),None,index)
            item['vertices'] = vertices
            item['joints'] = joints
            item['vertices_obj'] = None
        elif (self.options.SUPERVISE_DEPTH or self.options.SUPERVISE_SEGM) and '3dpw' not in self.dataset:
            if '3dpw-train-smplx' in self.dataset:
                cam_trans =self.cam_ext[index][:, 3].copy()
                cam_trans[:3] += self.trans_cam[index].copy()
            else:
                cam_trans =self.cam_ext[index][:, 3].copy()
                cam_trans[:3] += self.trans_cam[index].copy()
            vertices,joints = self.get_vertices(torch.tensor(self.pose_cam[index].copy()).float(),torch.tensor(self.betas[index].copy()).float(),cam_trans.copy(),index)
            item['vertices_obj'] = vertices.detach()
            item['joints']=joints.detach()
        imgname = os.path.join(self.img_dir, self.imgname[index])
        try:
            cv_img = read_img(imgname)
        except Exception as E:
            print(E)
            logger.info(f'@{imgname}@ from {self.dataset}')
        if self.is_train and 'closeup' in self.dataset:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)   
        
        img_w = cv_img.shape[1]
        img_h = cv_img.shape[0]
        #if self.is_train and (self.options.SUPERVISE_DEPTH or self.options.SUPERVISE_SEGM):
        #    part_segmentations = self.get_part_segmentation_maps(vertices.clone(),img_w,img_h,index)
        #    part_segmentations_copy = part_segmentations.copy()
        #print('SUCCESS'*10)
        if self.is_train and self.options.USE_DEPTH:
            if self.options.SUPERVISE_DEPTH and 'agora' not in self.dataset and '3dpw' not in self.dataset:
                depthname = os.path.join(self.depth_dir, self.depthname[index])
                maskname = os.path.join(self.mask_dir, self.maskname[index])
                try:
                    depth_map = read_depth_percentile_norm(depthname,maskname,keypoints_orig,'closeup' in self.dataset)
                    #parts_depth = read_human_parts_depth(depthname,part_segmentations,'closeup' in self.dataset)
                except Exception as E:
                    print(E)
                    logger.info(f'@{depthname}@ from {self.dataset}')
                    index = index - 1 if index > 0 else index + 1 
                    return self.__getitem__(index)
            else:
                depth_map = np.zeros((1,img_h,img_w))

        #if not self.is_train and self.options.USE_DEPTH and self.options.SUPERVISE_DEPTH:
        #    parts_depth = np.zeros((23,self.options.DEPTH_MAP_SIZE,self.options.DEPTH_MAP_SIZE))
        #if self.is_train and 'closeup' in self.dataset:
            #cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            '''          
            if self.options.USE_DEPTH and 'agora' not in self.dataset and '3dpw' not in self.dataset and self.options.SUPERVISE_DEPTH:
                #keypoints_orig_copy = np.rot90(keypoints_orig_copy, k=1, axes=(0, 1))

                import matplotlib.pyplot as plt
                plt.imshow(cv_img)
                plt.savefig('cv_img.png')
                plt.imshow(parts_depth[0])
                plt.savefig('parts_depth.png')
                import ipdb; ipdb.set_trace()
    
                parts_depth = np.rot90(parts_depth, k=1, axes=(2, 1))
            '''

        orig_shape = np.array(cv_img.shape)[:2]
        pose = self.pose_cam[index].copy()
        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale)
        if self.options.proj_verts:
            proj_verts = self.j2d_processing(proj_verts, center, sc * scale)
            item['proj_verts'] = torch.from_numpy(proj_verts).float()
       
        # Process image
        try:
            if self.options.USE_DEPTH and self.is_train and 'agora' not in self.dataset and '3dpw' not in self.dataset and self.options.SUPERVISE_DEPTH:
                img, depth_map,_ = self.rgb_depth_processing(cv_img, depth_map,None, center, sc*scale, kp2d=keypoints_orig_copy,
                                        img_res=self.options.IMG_RES)
                item['depth'] = torch.from_numpy(depth_map).float()          
            else:
                img = self.rgb_processing(cv_img, center, sc*scale, kp2d=keypoints,
                                      img_res=self.options.IMG_RES)
                item['depth'] = torch.zeros((1, self.options.DEPTH_MAP_SIZE, self.options.DEPTH_MAP_SIZE )).float()
        except Exception as E:
            logger.info(f'@{imgname} from {self.dataset}')
            print(E)

        try:
            img = torch.from_numpy(img).float()
        except Exception as E:
            logger.info(f'@{imgname} from {self.dataset}')
            print(E)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(self.betas[index]).float()
        item['imgname'] = imgname
        #has_depth boolean 25x1
        #item['has_depth'] = True
        if self.options.SUPERVISE_SEGM and self.is_train:
            gt_joints = joints.reshape(1, 127, 3)[:,:24]
            gt_keypoints_2d = keypoints.copy()
            gt_keypoints_2d=gt_keypoints_2d.reshape(1, 24, 3)
            gt_keypoints_2d[:, :, :-1] = \
            0.5 * self.options.IMG_RES * (gt_keypoints_2d[:, :, :-1] + 1)
            gt_keypoints_2d = torch.from_numpy(gt_keypoints_2d).float()

            gt_cam_t = estimate_translation(
                gt_joints.detach(),
                gt_keypoints_2d.detach(),
                focal_length=5000.0,
                img_size=224,
                use_all_joints=True,
            )
            item['parts_cam_t'] = gt_cam_t

        if self.is_train and self.options.USE_DEPTH and self.options.SUPERVISE_DEPTH:
            item['has_depth'] = True
            if depth_map.max() !=1 or 'agora' in self.dataset:
                item['has_depth'] = False

        #if self.is_train and self.options.SUPERVISE_DEPTH and item['has_depth'].sum() == 0:
            #index = index - 1 if index > 0 else index + 1 
            #return self.__getitem__(index)
        if self.is_train:
            if 'cam_int' in self.data.files:
                item['focal_length'] = torch.tensor([self.cam_int[index][0, 0], self.cam_int[index][1, 1]])
            if self.dataset == '3dpw-train-smplx':
                item['focal_length'] = torch.tensor([1961.1, 1969.2])
            # Will be 0 for 3dpw-train-smplx
            item['cam_ext'] = self.cam_ext[index]
            item['translation'] = self.cam_ext[index][:, 3]
            if 'trans_cam' in self.data.files:
                item['translation'][:3] += self.trans_cam[index]

        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        if self.is_train and '3dpw' not in self.dataset:
            if self.gender[index] == 1:
                faces = self.smplx_female.faces.copy()
            else:
                faces = self.smplx_male.faces.copy()
            item['faces'] = torch.tensor(faces.astype(np.int32))
        if self.options.ADVERSARIAL:
            if '3dpw' in self.dataset:
                item['adv_label'] = torch.tensor([1.0])
            else:
                item['adv_label'] = torch.tensor([0.0])

        if not self.is_train:
            item['dataset_index'] = self.options.VAL_DS.split('_').index(self.dataset)
        for k in item:
            if item[k] is None:
                item[k] = torch.tensor(0)
        return item

    def __len__(self):
        if self.is_train and 'agora' not in self.dataset and '3dpw' not in self.dataset:
            return int(self.options.CROP_PERCENT * len(self.imgname))
            #return len(self.imgname)
        else:
            return len(self.imgname)
