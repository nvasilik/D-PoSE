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
#from train.core.tester_smpl import Tester
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ["DISPLAY"] = ":0"e
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
def main(args):

    input_image_folder = args.image_folder
    output_path = args.output_folder
    #os.makedirs(output_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, 'demo.log'),
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

    tester = Tester(args)
    if True:
        all_image_folder = [input_image_folder]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_float32_matmul_precision('medium')
        with torch.cuda.amp.autocast(), torch.no_grad():
            mot = MPT(
                device=torch.device('cuda'),
                batch_size=4,
                display=False,
                detector_type='yolo',
                output_format='dict',
                yolo_img_size=416
            )
            cap = cv2.VideoCapture(2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS,13)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            frameNumber = 0
            use_bbox_filter = False
            while True:
                frameNumber+=1
                if True:#frameNumber%2==0:
                    ret, frame = cap.read()          
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
                    detection = mot.detector(input_tensor.cuda())

                    # Concatenate boxes and scores from all predictions at once
                    if detection:
                        #import ipdb; ipdb.set_trace()
                        t = torch.ones(4) * frameNumber
                        if use_bbox_filter:
                            boxes = torch.cat([bbox_one_euro_filter(t,pred['boxes']) for pred in detection], dim=0)
                        else:
                            boxes = torch.cat([pred['boxes'] for pred in detection], dim=0)
                        scores = torch.cat([pred['scores'] for pred in detection], dim=0)
                        # Apply threshold in a vectorized way
                        mask = scores > 0.7
                        # Filter and add scores as a new column
                        filtered_boxes = boxes[mask]
                        filtered_scores = scores[mask].unsqueeze(1)
                        # Merge boxes and scores using concatenation
                        dets = torch.cat([filtered_boxes, filtered_scores], dim=1).cpu().detach().numpy()
                    else:
                        dets = np.empty((0, 5))

                    detections = [dets]
                    detection = mot.prepare_output_detections(detections)
                    if len(detection[0]) > 0:

                        hmr_output=tester.run_on_single_image_tensor(frame, detection)
                        '''
                        from matplotlib import pyplot as plt
                        def get_smpl_skeleton():
                            return np.array(
                                [
                                    [ 0, 1 ],
                                    [ 0, 2 ],
                                    [ 0, 3 ],
                                    [ 1, 4 ],
                                    [ 2, 5 ],
                                    [ 3, 6 ],
                                    [ 4, 7 ],
                                    [ 5, 8 ],
                                    [ 6, 9 ],
                                    [ 7, 10],
                                    [ 8, 11],
                                    [ 9, 12],
                                    [ 9, 13],
                                    [ 9, 14],
                                    [12, 15],
                                    [13, 16],
                                    [14, 17],
                                    [16, 18],
                                    [17, 19],
                                    [18, 20],
                                    [19, 21],

                                ]
                            )
                        ax = None

                        def get_colors():
                            colors = {
                                'pink': np.array([197, 27, 125]),  # L lower leg
                                'light_pink': np.array([233, 163, 201]),  # L upper leg
                                'light_green': np.array([161, 215, 106]),  # L lower arm
                                'green': np.array([77, 146, 33]),  # L upper arm
                                'red': np.array([215, 48, 39]),  # head
                                'light_red': np.array([252, 146, 114]),  # head
                                'light_orange': np.array([252, 141, 89]),  # chest
                                'purple': np.array([118, 42, 131]),  # R lower leg
                                'light_purple': np.array([175, 141, 195]),  # R upper
                                'light_blue': np.array([145, 191, 219]),  # R lower arm
                                'blue': np.array([69, 117, 180]),  # R upper arm
                                'gray': np.array([130, 130, 130]),  #
                                'white': np.array([255, 255, 255]),  #
                                'pinkish': np.array([204, 77, 77]),
                            }
                            return colors
                        radius =1
                        joints = joints[:len(get_smpl_skeleton())+1]
                        if True:
                            fig = plt.figure(figsize=(12, 7))
                            ax = fig.add_subplot(111, projection='3d')
                            ax.set_aspect('auto')
                        skeleton = get_smpl_skeleton()
                        kp_3d = joints
                        for i, (j1, j2) in enumerate(skeleton):
                                if kp_3d[j1].shape[0] == 4:
                                    x, y, z, v = [np.array([kp_3d[j1, c], kp_3d[j2, c]]) for c in range(4)]
                                else:
                                    x, y, z = [np.array([kp_3d[j1, c], kp_3d[j2, c]]) for c in range(3)]
                                    v = [1, 1]
                                ax.plot(x, y, z, lw=2, c=get_colors()['purple'] / 255)
                                for j in range(2):
                                    if v[j] > 0: # if visible
                                        ax.plot(x[j], y[j], z[j], lw=2, c=get_colors()['blue'] / 255, marker='o')
                                    else: # nonvisible
                                        ax.plot(x[j], y[j], z[j], lw=2, c=get_colors()['red'] / 255, marker='x')

                        pelvis_joint = 0
                        RADIUS = radius  # space around the subject
                        xroot, yroot, zroot = kp_3d[pelvis_joint, 0], kp_3d[pelvis_joint, 1], kp_3d[pelvis_joint, 2]
                        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
                        ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                        ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("z")
                        ax.view_init(-90, -90)
                        if ax is None:
                            plt.show()
                        plt.savefig('skeleton.png')
                        import ipdb; ipdb.set_trace()
                        '''

                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('front', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    del tester.model

    logger.info('================= END =================')


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

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default='data/ssp_3d_test.npz')
    parser.add_argument('--data_split', type=str, default='test')

    args = parser.parse_args()
    main(args)
