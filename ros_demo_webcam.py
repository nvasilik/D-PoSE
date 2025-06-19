import os
import sys
import argparse
import torch
import numpy as np
import cv2
from loguru import logger
from glob import glob
from train.core.tester import Tester
from train.utils.one_euro_filter import OneEuroFilter
from multi_person_tracker import MPT
from multi_person_tracker import Sort
#Dataloader
from torch.utils.data import DataLoader
from skeleton_msgs.msg import Skeletons, Skeleton, Joint3D#from train.core.tester_smpl import Tester
from aruco.aruco_create import detect_aruco_from_image
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ["DISPLAY"] = ":0"e
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf_transformations
sys.path.append('')
'''
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d

def smooth_bbox_params(bbox_params, kernel_size=11, sigma=8):
    """
    Applies median filtering and then gaussian filtering to bounding box
    parameters.

    Args:
        bbox_params (torch.Tensor or np.ndarray): Shape (N, 4) with [x1, y1, x2, y2].
        kernel_size (int): Kernel size for median filtering (must be odd).
        sigma (float): Sigma for gaussian smoothing.

    Returns:
        torch.Tensor: Smoothed bounding box parameters (N, 4).
    """
    if isinstance(bbox_params, torch.Tensor):
        bbox_params = bbox_params.cpu().numpy()  # Convert to NumPy for processing

    # Ensure we have at least kernel_size elements to avoid zero-padding warning
    if bbox_params.shape[0] < kernel_size:
        kernel_size = max(1, bbox_params.shape[0] // 2 * 2 + 1)  # Keep kernel size odd

    # Apply median and Gaussian filtering
    smoothed = np.array([signal.medfilt(param, kernel_size) for param in bbox_params.T]).T
    smoothed = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T

    return torch.tensor(smoothed, dtype=torch.float32, device='cpu') 
'''

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
def rotmat_to_quat(R):
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    assert R.shape == (3, 3)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])

import math

def degrees_to_radians(deg):
    return math.radians(deg)
class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Skeletons, 'humans', 10)

        logger.add(
            os.path.join('.', 'demo.log'),
            level='INFO',
            colorize=False,
        )
        logger.info(f'Demo options: \n {args}')
        bbox_one_euro_filter = OneEuroFilter(
            np.zeros(4),
            np.zeros(4),
            min_cutoff=0.004,
            beta=0.4,
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        tester = Tester(args)
        first_rvec= None
        first_tvec = None
        if True:
            #all_image_folder = [input_image_folder]
            #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            torch.set_float32_matmul_precision('medium')
            with torch.cuda.amp.autocast(), torch.no_grad():
                mot = MPT(
                    device=torch.device('cuda'),
                    batch_size=4,
                    display=False,
                    detector_type='maskrcnn',
                    output_format='list',
                    yolo_img_size=256
                )
                cap = cv2.VideoCapture(4)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS,13)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                frameNumber = 0
                use_bbox_filter = False
                self.tracker = Sort()
                while True:
                    frameNumber+=1
                    if True:#frameNumber%2==0:
                        ret, frame = cap.read()  
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #Flip frame horizontally
                        

                        input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
                        detection = mot.detector(input_tensor.cuda())
                        #if first_rvec is None or first_tvec is None:
                        first_rvec, first_tvec = detect_aruco_from_image(frame)
                        # Concatenate boxes and scores from all predictions at once
                        if detection:
                            #import ipdb; ipdb.set_trace()
                            t = torch.ones(4) * frameNumber
                            if use_bbox_filter:
                                boxes = torch.cat([bbox_one_euro_filter(t,pred['boxes']) for pred in detection], dim=0)
                            else:
                                boxes = torch.cat([pred['boxes'] for pred in detection], dim=0)
                                #import ipdb; ipdb.set_trace()
                            scores = torch.cat([pred['scores'] for pred in detection], dim=0)
                            # Apply threshold in a vectorized way
                            mask = scores > 0.7
                            # Filter and add scores as a new column
                            filtered_boxes = boxes[mask]
                            filtered_scores = scores[mask].unsqueeze(1)
                            # Merge boxes and scores using concatenation

                            dets = torch.cat([filtered_boxes, filtered_scores], dim=1).cpu().detach().numpy()
                            #import ipdb; ipdb.set_trace()
                            if dets.shape[0] > 0:
                                track_bbs_ids = self.tracker.update(dets)
                                # Update tracker with the detections
                        else:
                            track_bbs_ids = np.empty((0, 5))
                            dets = np.empty((0, 5))

                        #print('BBs ids:', track_bbs_ids[:, -1])
                        #import ipdb; ipdb.set_trace()
                        detections = [dets]
                        detection = mot.prepare_output_detections(detections)
                    if len(detection[0]) > 0:


                        hmr_output=tester.run_on_single_image_tensor(frame, detection)
                        #import ipdb; ipdb.set_trace()
                        Skeletons_ros = Skeletons()
                        Human = Skeleton()
                        Skeletons_ros.humans = []

                        #Joint3D[] joints  # Array of joints for this skeleton (one frame)
                        #uint32 id  # Unique ID for the human

                        #Joints3D:
                        #float64 x
                        #float64 y
                        #float64 z
                        #Joints3D = hmr_output['joints3d']
                        hmr_joints = hmr_output['joints3d'][:,0:22,:].cpu().numpy()  # Assuming hmr_output['joints3d'] is a tensor of shape (N, 21, 3)
                        camera_translation = hmr_output['pred_cam_t'].cpu().numpy()  # Assuming this is a tensor of shape (N, 3)
                        camera_translation= camera_translation*0.5  # Scale the translation if needed
                        now = self.get_clock().now().to_msg()
                        for i in range(len(track_bbs_ids)):
                            Human = Skeleton()  # <-- Moved this line inside the loop
                            joints = hmr_joints[i]#tvec.reshape(-1,3)
                            #Apply camera translation to joints
                            joints[:, 0] += camera_translation[i, 0]
                            joints[:, 1] += camera_translation[i, 1]
                            joints[:, 2] += camera_translation[i, 2]
                            Human.joints = []
                            for j,joint in enumerate(joints):
                                joint3d = Joint3D()
                                joint3d.x = float(joint[0])
                                joint3d.y = float(joint[1])
                                joint3d.z = float(joint[2])
                                Human.joints.append(joint3d)
                                if j==0:
                                    #print('Joint 0:', joint3d.x, joint3d.y, joint3d.z)
                                    t = TransformStamped()
                                    t.header.stamp = now
                                    t.header.frame_id = 'Camera'  # Or 'camera_link' or whatever your base frame is
                                    t.child_frame_id = f'human_{int(track_bbs_ids[i][-1])}_joint_{j}'
                                    t.transform.translation.x = joint3d.x
                                    t.transform.translation.y = joint3d.y
                                    t.transform.translation.z = joint3d.z
                                    t.transform.rotation.x = -1.0
                                    t.transform.rotation.y = 0.0
                                    t.transform.rotation.z = 0.0
                                    t.transform.rotation.w = 1.0
                                    self.tf_broadcaster.sendTransform(t)
                                Human.id = int(track_bbs_ids[i][-1])
                                Skeletons_ros.humans.append(Human)

                                #Broadcast rvec and tvec as a transform
                        #if first_rvec is not None and first_tvec is not None:
                        from tf_transformations import quaternion_about_axis, quaternion_multiply

                        t = TransformStamped()
                        t.header.stamp = now
                        t.header.frame_id = 'base_link'
                        t.child_frame_id = f'Aruco_marker'
                        t.transform.translation.x = 0.0#float(tvec[0])
                        t.transform.translation.y = -0.15#float(tvec[1])
                        t.transform.translation.z = -0.2#float(tvec[2])

                        t.transform.rotation.x = 0.7071068
                        t.transform.rotation.y = 0.0
                        t.transform.rotation.z = 0.0
                        t.transform.rotation.w =  0.7071068
                        self.tf_broadcaster.sendTransform(t)

                        if first_rvec is not None and first_tvec is not None:
                            t = TransformStamped()
                            t.header.stamp = now
                            t.header.frame_id = 'Aruco_marker'
                            t.child_frame_id = f'Camera'
                            trans = first_tvec

                            #Align the camera with the Aruco marker
                            t.transform.translation.x = float(trans[0])
                            t.transform.translation.y = float(-trans[1])
                            t.transform.translation.z = float(trans[2])
                            rvec = first_rvec
                            
                                                    
                                                    
                            # Convert rotation vector to rotation matrix
                            R = cv2.Rodrigues(first_rvec)[0]

                            # Invert rotation: R_inv = R.T
                            R_inv = R.T

                            # Invert translation: t_inv = -R.T @ t
                            tvec = first_tvec.reshape(3)
                            t_inv = -np.dot(R_inv, tvec)

                            # Fill in the inverted translation
                            t.transform.translation.x = float(-t_inv[0])
                            t.transform.translation.y = float(t_inv[1])
                            t.transform.translation.z = float(t_inv[2])

                            # Convert inverted rotation matrix back to quaternion
                            quat = rotmat_to_quat(R_inv)
                            t.transform.rotation.x = float(quat[1])
                            t.transform.rotation.y = float(quat[2])
                            t.transform.rotation.z = float(quat[3])
                            t.transform.rotation.w = float(quat[0])


                            self.tf_broadcaster.sendTransform(t)
                                # Broadcast transform for each joint
                        #import ipdb; ipdb.set_trace()
                        self.publisher_.publish(Skeletons_ros)

                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('front', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        del tester.model

        logger.info('================= END =================')
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #elf.i = 0

        
        #self.get_logger().info('Publishing: "%s"' % msg.data)
@torch.no_grad()
def main(args):
    rclpy.init()
    ros = MinimalPublisher()
    rclpy.spin(ros)
    #input_image_folder = args.image_folder
    #output_path = args.output_folder
    #os.makedirs(output_path, exist_ok=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/dpose_conf.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/paper_arxiv.ckpt',
                        help='checkpoint path')

    parser.add_argument('--image_folder', type=str, default='demo_images',
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='demo_images/results',
                        help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')
                        
    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=256,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default='data/ssp_3d_test.npz')
    parser.add_argument('--data_split', type=str, default='test')

    args = parser.parse_args()
    main(args)
