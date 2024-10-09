import torch
import smplx
import pickle
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from . import constants
from . import config
from .constants import NUM_JOINTS_SMPLX
from ..dataset.dataset import DatasetHMR
from ..utils.eval_utils import reconstruction_error
from ..utils.renderer import Renderer
from ..utils.renderer_pyrd import Renderer as RendererPyrd
from ..models.hmr import HMR
from ..losses.losses import HMRLoss

class HMRTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(HMRTrainer, self).__init__()

        self.hparams.update(hparams)
        self.model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        )

        self.loss_fn = HMRLoss(hparams=self.hparams)

        self.smplx = smplx.SMPLX(config.SMPLX_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.smpl = smplx.SMPL(config.SMPL_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False)

        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()

        self.val_ds = self.val_dataset()
        self.save_itr = 0
        self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32).cuda()

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smplx.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

    def forward(self, x, bbox_center, bbox_scale, img_w, img_h, fl=None):
        return self.model(x, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h, fl=fl) 

    def training_step(self, batch, batch_nb, dataloader_nb=0):
        # GT data
        images = batch['img']
        gt_betas = batch['betas']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        fl = batch['focal_length']
        gt_pose = batch['pose']
        batch_size = batch['img'].shape[0]

        pred, depth,_,_, part_segms = self(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h, fl=fl)
        pred['depth'] = depth
        pred['part_segms'] = part_segms
        # Calculate joints and vertices using just 22 pose param for SMPL
        gt_out = self.smplx(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:NUM_JOINTS_SMPLX*3],
            global_orient=gt_pose[:, :3]
        )
        batch['vertices'] = gt_out.vertices
        batch['joints3d'] = gt_out.joints

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        self.log('train_loss', loss, logger=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)

        return {'loss': loss}

    
    def validation_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):

        images = batch['img']
        batch_size = images.shape[0]
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        dataset_names = batch['dataset_name']
        dataset_index = batch['dataset_index'].detach().cpu().numpy()
        val_dataset_names = self.hparams.DATASET.VAL_DS.split('_')
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        J_regressor_batch_smpl = self.J_regressor[None, :].expand(batch['img'].shape[0], -1, -1)
        #if not self.hparams.DATASET.USE_DEPTH:
        #    pred,_,_,_,_ = self(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
        #else:
        pred, depth,_,_,part_segms = self(images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
        pred['part_segms'] = part_segms
        pred['depth'] = depth
        pred_cam_vertices = pred['vertices']

        joint_mapper_gt = constants.J24_TO_J14
        joint_mapper_h36m = constants.H36M_TO_J14

        if 'bedlam' in dataset_names[0]:
            gt_out_cam = self.smplx(
                betas=batch['betas'],
                body_pose=batch['pose'][:, 3:NUM_JOINTS_SMPLX*3],
                global_orient=batch['pose'][:, :3],
            )
            gt_cam_vertices = gt_out_cam.vertices
            gt_keypoints_3d = gt_out_cam.joints[:, :24]
            pred_keypoints_3d = pred['joints3d'][:, :24]
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0

            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
        elif 'rich' in dataset_names[0]:
            # For rich vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            gt_keypoints_3d = batch['joints']
            pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1), pred_cam_vertices)
            pred_keypoints_3d = torch.matmul(self.smpl.J_regressor, pred_cam_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
        elif 'h36m' in dataset_names[0]:
            gt_cam_vertices = batch['vertices']
            # # Get 14 predicted joints from the mesh
            gt_keypoints_3d = batch['joints']
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            gt_keypoints_3d = gt_keypoints_3d - ((gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)
            pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1).cuda(), pred_cam_vertices)
            # # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            # pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_keypoints_3d = pred_keypoints_3d - ((pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)
        else:
            # For 3dpw vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            # Get 14 predicted joints from the mesh
            gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_cam_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            # Convert predicted vertices to SMPL Fromat
            pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1), pred_cam_vertices)
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
        error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1)).cpu().numpy()

        # Reconstuction_error (PA-MPJPE)
        r_error, _ = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None
        )
        val_mpjpe = error.mean(-1)
        val_pampjpe = r_error.mean(-1)
        val_pve = error_verts.mean(-1)

        loss_dict = {}

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            ds_idx = val_dataset_names.index(ds.dataset)
            idxs = np.where(dataset_index == ds_idx)
            loss_dict[ds_name + '_mpjpe'] = list(val_mpjpe[idxs])
            loss_dict[ds_name + '_pampjpe'] = list(val_pampjpe[idxs])
            loss_dict[ds_name + '_pve'] = list(val_pve[idxs])

        return loss_dict

    def validation_epoch_end(self, outputs):
        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}

        if len(self.val_ds) > 1:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(np.array([val[ds_name + '_mpjpe'] for x in outputs for val in x])).mean()
                pampjpe = 1000 * np.hstack(np.array([val[ds_name + '_pampjpe'] for x in outputs for val in x])).mean()
                pve = 1000 * np.hstack(np.array([val[ds_name + '_pve'] for x in outputs for val in x])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))
                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve
        else:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
                pampjpe = 1000 * np.hstack(np.array([x[ds_name + '_pampjpe'] for x in outputs])).mean()
                pve = 1000 * np.hstack(np.array([x[ds_name + '_pve'] for x in outputs])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))

                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve

        self.log('val_loss', val_log[self.val_ds[0].dataset + '_val_pampjpe'], logger=True, sync_dist=True)
        self.log('val_loss_mpjpe', val_log[self.val_ds[0].dataset + '_val_mpjpe'], logger=True, sync_dist=True)
        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True)

    
    def test_step(self, batch, batch_nb, dataloader_nb=0):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):

        return torch.optim.Adam(
                        self.parameters(),
                        lr=self.hparams.OPTIMIZER.LR,
                        weight_decay=self.hparams.OPTIMIZER.WD)


    def train_dataset(self):
        options = self.hparams.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_list = [DatasetHMR(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)


        return train_ds
    
    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        img_dataloader = DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
            drop_last=True
        )

        return img_dataloader

    def val_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                DatasetHMR(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    is_train=False,
                )
            )
        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                    drop_last=True
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()
