"""
aioz.aiar.truongle - Jun 02, 2020

"""
import cv2
import os
import glob
import argparse
import numpy as np
from config import Config
from inference.utils import utils
from inference.api.face_detection_api import FaceDetectionApi
from inference.api.face_ft_extractor_api import FaceFtExtractorApi
from inference.models.wrapper.gender_detection_efficent_net import GenderDetectionPb
from inference.models.wrapper.emotion_efficent_net import ExpressionDetectionPb

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input_dir", type=str, default="data/db_video/USHER")
parse.add_argument("-o", "--output_dir", type=str, default="data/test_video/hiv00004_res")
args = parse.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
OUT_VIDEO_PATH = "data/test_video/hiv00004_res_.mp4"
STEP = 10

params = {'FILTER_BOX_SIZE': 20,
          'RESIZE_FRAME_RATIO': 1,
          'HEAD_POSE_THRESH': 30,
          'FACE_SIZE_OUTPUT': 112}


def vis(image, bbox, name, gen, emo, color=(0, 0, 255)):
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    color = color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    # draw bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    # Draw name
    label = name
    text_size = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(image,
                  (xmin, ymin - 10 - text_size[0][1]),
                  (xmin + 10 + text_size[0][0], ymin),
                  color, -1)
    cv2.putText(image, label,
                (xmin + 5, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)

    text_size = cv2.getTextSize(gen, font, font_scale, thickness)
    txt_loc = (xmin, ymax + 10 + text_size[0][1])
    cv2.putText(image,
                text=gen,
                org=txt_loc,
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness)

    # EXPR
    text_size = cv2.getTextSize(emo, font, font_scale, thickness)
    txt_loc = (xmin, ymax + 30 + 2*text_size[0][1])
    cv2.putText(image,
                text=emo,
                org=txt_loc,
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = Config()
    face_detection_api = FaceDetectionApi(config=config, params=params)
    print('\nhere\n')
    face_ft_extractor = FaceFtExtractorApi(config=config)
    print('\nhere\n')
    gender_estimator = GenderDetectionPb(config=config)
    emo_estimation = ExpressionDetectionPb(config=config)


    face_sample = "data/test_video/USHER - DEA_000030.jpg"
    # face_sample = "data/test_video/sample.jpg"
    
    video_pth = "data/test_video/hiv00004.mp4"
    cap = cv2.VideoCapture(video_pth)
    vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # INIT WRITER
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (int(vid_w), int(vid_h)))
    print("[INFO] information:  ", vid_w, vid_h, fps)
    count = 0
    
    # GET SAMPLE:
    img_sample = cv2.imread(face_sample)
    img_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
    img_sample = np.transpose(img_sample, (2, 0, 1))
    face_ft_sample = face_ft_extractor.proceed(img_sample)
    emo_s = np.zeros(2)
    list_emo = ["Hap", "Sad"]

    # Feature list
    feature_list = []
    
    while cap.isOpened() and count <= 180*30:
        ret, frame = cap.read()
        if ret:
            # DETECTION
            boxes, faces, elapsed = face_detection_api.proceed(frame=frame, vid_w=vid_w, vid_h=vid_h)
            if boxes is not None:
                sim = []
                for box, face in zip(boxes, faces):
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = np.transpose(face, (2, 0, 1))
                    face_ft = face_ft_extractor.proceed(face)
                    _sim = np.dot(face_ft_sample, face_ft.T)
                    sim.append(_sim)
                sim = np.asarray(sim)
                indices = np.argmax(sim)
                print(sim[indices])
                if sim[indices] > 0.3:
                    face = faces[indices]
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                    emo = emo_estimation.process_prediction(face)[0]
                    if emo in list_emo:
                        idx = list_emo.index(emo)
                        emo_s[idx] += 1
                    emo_percent = emo_s / np.sum(emo_s)
                    emo_percent = np.around(emo_percent, decimals=2)
                    print(emo_s)
                    if count > 10:
                        emo_str = "Pos: %s, Neg: %s" % (emo_percent[0], emo_percent[1])
                    else:
                        emo_str = ''
                    gen = gender_estimator.process_prediction(face, use_tta=True)
                    gen_str = "M" if gen > 0.5 else "F"
                    vis(image=frame, bbox=boxes[indices], name="DAE", gen=gen_str, emo=emo_str)
            print("Frame: {}, Time: {}".format(count, elapsed))
            out.write(frame)
            frame = cv2.resize(frame, (int(vid_w//3), int(vid_h//3)))
            cv2.imshow("abc", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    main()



