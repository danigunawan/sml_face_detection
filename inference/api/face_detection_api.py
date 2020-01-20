"""
aioz.aiar.truongle - Jan 02, 2020
face detection api: face detection, filter with face size, filter with head pose
"""
import cv2
import gc
import time
import numpy as np
import keras
import tensorflow as tf
from termcolor import colored
from ..utils import utils
from ..utils import tensorflow_utils as tf_utils
from ..models.wrapper.yolo_face_wrapper import YoloFace
from ..models.wrapper.headpose_inference_fsanet import HeadPoseDetection


class FaceDetectionApi:
    def __init__(self, config, params=None):
        """align_mode include 3 mode: face_al, dlib, non
        """
        # INIT PARAMETER
        if params is None:
            params = {}
        self.resize_frame_ratio = utils.check_param('RESIZE_FRAME_RATIO', params=params, default_value=1)
        self.filter_box_size = utils.check_param('FILTER_BOX_SIZE', params=params, default_value=None)
        self.head_pose_thresh = utils.check_param('HEAD_POSE_THRESH', params=params, default_value=None)
        self.face_size_output = utils.check_param('FACE_SIZE_OUTPUT', params=params, default_value=112)
        self.crop_face_margin = (0.1, 0.1, 0.1, 0.1)  # top-left-bottom-right
        self.config = config
        print(colored("[INFO] Clear session ... ", color='red', attrs=['bold']))
        self._clear_sess()
        print(colored("[INFO] Load model ... ", color='cyan', attrs=['bold']))
        self._init_all_model()
        print(colored("[INFO] Load model is DONE... ", color='cyan', attrs=['bold']))

    def _init_all_model(self):
        # FACE DETECTOR
        # self.face_detection = FaceDetection(self.config)
        self.face_detection = YoloFace(self.config)
        self.face_cropper = tf_utils.ImageCropper()  # crop + resize with tensor-flow
        self.face_resize = tf_utils.ImageResize()  # resize with tensor-flow

        # HEAD POSE
        self.head_pose_detection = HeadPoseDetection(self.config)

    @staticmethod
    def _clear_sess():
        """
        Clear all tf and keras session
        --
        clear_session(): Destroys the current TF graph and creates a new one.
        Useful to avoid clutter from old models / layers.
        """
        gc.collect()
        tf.reset_default_graph()
        sess = tf.get_default_session()
        sess = None
        keras.backend.clear_session()
        tf.keras.backend.clear_session()

    def proceed(self, frame, vid_w, vid_h):
        start = time.time()
        rescaled_frame = cv2.resize(frame, (int(vid_w / self.resize_frame_ratio), int(vid_h / self.resize_frame_ratio)))
        boxes = self.face_detection.process_detection(rescaled_frame, filter_size=self.filter_box_size)

        if len(boxes) > 0:
            boxes = boxes * self.resize_frame_ratio
            # HEAD POSE & FILTER
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            faces_64_64 = self.face_cropper.tf_run_crop(frame, boxes, crop_size=(64, 64))  # crop by tensorflow
            head_poses, head_pose_mask = self.head_pose_detection.process_prediction(faces_64_64,
                                                                                     threshold=self.head_pose_thresh)
            # DRAW HEAD POSE
            # utils.draw_head_pose(img=frame_ori, boxes=boxes, head_poses=head_poses, size=15)
            boxes = boxes[head_pose_mask]
            faces_crop = self.face_cropper.tf_run_crop(frame, boxes,
                                                       crop_size=(self.face_size_output, self.face_size_output),
                                                       margin=self.crop_face_margin)
        if len(boxes) > 0:
            elapsed = time.time() - start
            return boxes, faces_crop, elapsed
        else:
            elapsed = time.time() - start
            return None, None, elapsed
