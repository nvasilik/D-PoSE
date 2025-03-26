import os
import sys
import argparse
import torch
import numpy as np
import cv2
from loguru import logger
from glob import glob
from train.core.tester import Tester
from multi_person_tracker import MPT
from multi_person_tracker import Sort
#Dataloader
from torch.utils.data import DataLoader
#from train.core.tester_smpl import Tester
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ["DISPLAY"] = ":0"e
sys.path.append('')


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

    tester = Tester(args)
    if True:
        all_image_folder = [input_image_folder]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        mot = MPT(
            device=device,
            batch_size=1,
            display=False,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416
        )

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS,20)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frameNumber = 0
        while True:
          frameNumber+=1
          if True:#frameNumber%2==0:
            ret, frame = cap.read()          
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
            detection = mot.detector(input_tensor)

            # Concatenate boxes and scores from all predictions at once
            if detection:
                boxes = torch.cat([pred['boxes'] for pred in detection], dim=0)
                scores = torch.cat([pred['scores'] for pred in detection], dim=0)
                # Apply threshold in a vectorized way
                mask = scores > 0.7
                # Filter and add scores as a new column
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask].unsqueeze(1)
                # Merge boxes and scores using concatenation
                dets = torch.cat([filtered_boxes, filtered_scores], dim=1).cpu().numpy()
            else:
                dets = np.empty((0, 5))

            detections = [dets]
            detection = mot.prepare_output_detections(detections)
            if len(detection[0]) > 0:

                tester.run_on_single_image_tensor(frame, detection)

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

    parser.add_argument('--detector', type=str, default='maskrcnn', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default='data/ssp_3d_test.npz')
    parser.add_argument('--data_split', type=str, default='test')

    args = parser.parse_args()
    main(args)
