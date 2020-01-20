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
from face_list import Face_List

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

def detect_face_information(face, emo_estimation, gender_estimator):
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    emo = emo_estimation.process_prediction(face)[0]
    gen = gender_estimator.process_prediction(face, use_tta=True)
    return emo, gen

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = Config()
    face_detection_api = FaceDetectionApi(config=config, params=params)
    face_ft_extractor = FaceFtExtractorApi(config=config)
    gender_estimator = GenderDetectionPb(config=config)
    emo_estimation = ExpressionDetectionPb(config=config)


    face_sample = "data/test_video/USHER - DEA_000030.jpg"
    # face_sample = "data/test_video/sample.jpg"
    
    #video_pth = "data/test_video/hiv00004.mp4"
    cam = "cam02"
    name = "191215_014.mp4"
    video_pth = "/media/aioz-trung-intern/data/sml/" + cam + "/" + name
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

    # Face list
    face_list = Face_List()

    # Load mask
    mask_name = "mask_cam02.jpg"
    mask = cv2.imread(mask_name, 0)

    is_update = False

    while cap.isOpened and count <= 600*30:
        ret, frame = cap.read()
        if ret:
            # CROP frame
            org_frame = frame.copy()
            frame = cv2.bitwise_and(frame, frame, mask=mask)

            # DETECTION
            boxes, faces, elapsed = face_detection_api.proceed(frame=frame, vid_w=vid_w, vid_h=vid_h)
            if boxes is not None:
                sim = []
                for box, face in zip(boxes, faces):
                    # Detect face gender and emotion
                    org_face = face.copy()
                    emo, gen = detect_face_information(org_face, emo_estimation, gender_estimator)
                    # Print face gender and emotion
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
                    # gen = gender_estimator.process_prediction(face, use_tta=True)
                    gen_str = "M" if gen > 0.5 else "F"
                    vis(image=org_frame, bbox=box, name="DAE", gen=gen_str, emo=emo_str)

                    # Extract face's feature
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = np.transpose(face, (2, 0, 1))
                    face_ft = face_ft_extractor.proceed(face)

                    # Check new face
                    flag = False
                    for i in range(face_list.num_of_face):
                        # Calculating face similarity
                        _sim = np.dot(face_list.feature_container[i], face_ft.T)
                        if (_sim >= 0.2):
                            flag=True
                            face_list.duration_container[i].append(count)
                            face_list.emotion_container[i].append(emo)
                            break

                    if (not flag):
                        face_list.add_new_face(org_face, face_ft, emo, count)
                    
                    is_update = True

                # CALCULATING similarity
                #     _sim = np.dot(face_ft_sample, face_ft.T)
                #     sim.append(_sim)
                # sim = np.asarray(sim)
                # indices = np.argmax(sim)
                # print(sim[indices])

                # FACE REGCONITION, GENDER and EMOTION DETECTION
                # face = faces[0]
                # face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                # face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                # emo = emo_estimation.process_prediction(face)[0]
                
            print("Frame: {}, Time: {}".format(count, elapsed))
            out.write(org_frame)
            org_frame = cv2.resize(org_frame, (int(vid_w//2), int(vid_h//2)))
            cv2.imshow("abc", org_frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    print(face_list.emotion_container)
    face_list.save_list()


if __name__ == '__main__':
    main()



