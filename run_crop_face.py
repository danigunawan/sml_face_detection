"""
aioz.aiar.truongle - Dec 30, 2019
build database for face search
"""
import cv2
import os
import glob
import argparse
from config import Config
from inference.api.face_detection_api import FaceDetectionApi

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input_dir", type=str, default="data/db_video/USHER")
parse.add_argument("-o", "--output_dir", type=str, default="data/db_video/face_crop")
args = parse.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
STEP = 10

params = {'FILTER_BOX_SIZE': 20,
          'RESIZE_FRAME_RATIO': 1,
          'HEAD_POSE_THRESH': 30,
          'FACE_SIZE_OUTPUT': 112}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = Config()
    face_detection_api = FaceDetectionApi(config=config, params=params)
    cam = "cam02"
    name = "191215_014.mp4"
    video_pth = "/media/aioz-trung-intern/data/sml/" + cam + "/" + name
    list_video = [video_pth] #glob.glob("%s/*.mp4" % INPUT_DIR)
    for video_pth in list_video:
        vid_name = os.path.split(video_pth)[-1].split(".")[0]
        cap = cv2.VideoCapture(video_pth)
        vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        center = (vid_w/2, vid_h/2)
        print("[INFO] information:  ", vid_w, vid_h)
        count = 0
        while cap.isOpened() and count <= total_frame:
            ret, frame = cap.read()
            if ret:
                # ROTATE
                # M = cv2.getRotationMatrix2D(center, angle=270, scale=1.0)
                # frame = cv2.warpAffine(frame, M, (int(vid_w), int(vid_h)))
                if count % STEP == 0:
                    # DETECTION
                    boxes, faces, elapsed = face_detection_api.proceed(frame=frame, vid_w=vid_w, vid_h=vid_h)
                    if faces is not None:
                        for num, f in enumerate(faces):
                            cv2.imwrite(os.path.join(OUTPUT_DIR, vid_name + "_%.6d_%.3d.jpg" % (count, num)), f)

                frame = cv2.resize(frame, (int(vid_w//3), int(vid_h//3)))
                cv2.imshow("abc", frame)
            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()


if __name__ == '__main__':
    main()



