import numpy as np
import sys
import os
import cv2
import math
from math import cos, sin
from termcolor import colored
import colorsys


class CropFrame:
    def __init__(self, vid_w, vid_h, tlbr_ratio=None):
        """tlbr_ratio: ratio for top, left, bottom, right"""
        if tlbr_ratio is None:
            tlbr_ratio = [0.2, 0.2, 0.2, 0.2]
        self.xmin_crop = int(tlbr_ratio[1] * vid_w)
        self.ymin_crop = int(tlbr_ratio[0] * vid_h)
        self.xmax_crop = int(vid_w - tlbr_ratio[3] * vid_w)
        self.ymax_crop = int(vid_h - tlbr_ratio[2] * vid_h)
        self.w_crop = self.xmax_crop - self.xmin_crop
        self.h_crop = self.ymax_crop - self.ymin_crop

    def crop(self, frame):
        crop_frame = frame[self.ymin_crop: self.ymax_crop, self.xmin_crop: self.xmax_crop, :]
        return crop_frame


def check_param(key, params, default_value):
    """
    :param key: key need to check
    :param params: dictionary store parameter
    :param default_value: default value for key
    :return:
    """
    value = params[key] if key in params.keys() else default_value
    return value


def adjust_lightness(image, gamma=2.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def check_dir(directory):
    if os.path.isdir(directory):
        print(colored("[INFO] {} is exist".format(directory), "cyan", attrs=['bold']))
    else:
        os.mkdir(directory)
        print(colored("[INFO] created {}".format(directory), "cyan", attrs=['bold']))


def check_file(file_path):
    if not os.path.isfile(file_path):
        print("File path {} does not exist. Exiting...".format(file_path))
        sys.exit()


def calculate_mean_std_track(tracks):
    list_id = []
    list_mean = []
    list_std = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        std = np.std(track.ages)
        mean = np.mean(track.ages)
        list_id.append(track.track_id)
        list_std.append(std)
        list_mean.append(mean)

    return list_id, list_mean, list_std


def convert_type(box):
    """
    :param box: boxes x, y, w, h
    :return: boxes xmin, ymin, xmax, ymax
    """
    new_box = np.copy(box)
    new_box[:, 2] = box[:, 0] + box[:, 2]
    new_box[:, 3] = box[:, 1] + box[:, 3]

    return new_box


def write_xml_file(data, path):
    """
    :param data: data for write
    :param path: path for saved file
    :return: done
    """
    with open(path, 'w') as f_write:
        f_write.write(data)
    f_write.close()
    print(colored('[INFO] write file success at {}'.format(path), 'yellow', attrs=['bold']))


def write_img(img, path, print_flat=True):
    """
    :param img: img need to save
    :param path: path for save
    :return: done
    """
    cv2.imwrite(path, img)
    if print_flat:
        print('[INFO] write image success at {}'.format(path))


def create_unique_color(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    color = (int(255 * r), int(255 * g), int(255 * b))
    return color


def resize_batch(imgs, dim):
    resized_imgs = []
    for img in imgs:
        resized_imgs.append(cv2.resize(img, dim))
    return np.array(resized_imgs)


def draw_bbox(img, bboxes, color_bbox=(0, 255, 255)):
    """
    :param img: images need ro draw box
    :param bboxes: bounding box, format [xmin, ymin, xmax, ymax]
    :param color_bbox: color of bounding box
    :return: images has been drawn box
    """
    bboxes = np.asarray(bboxes)
    if np.ndim(bboxes) == 1:
        bboxes = np.expand_dims(bboxes, axis=0)

    for i, bbox in enumerate(bboxes):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, 2)


def draw_single_bbox_with_info(img, bbox, id, info, color_bbox=(0, 255, 255)):
    color = create_unique_color(id)

    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    txt_loc = (xmin, ymax + 20)
    cv2.putText(img,
                text="ID: " + str(id),
                org=txt_loc,
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness)

    txt_loc = (xmin, ymax + 40)
    cv2.putText(img,
                text=str(info),
                org=txt_loc,
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=thickness)


def draw_bbox_with_validation_info(img, bboxes, info, color_bbox=(0, 255, 255)):
    for i, bbox in enumerate(bboxes):
        if np.array_equal(bbox, [0.0, 0.0, 0.0, 0.0]) == False:
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, 2)

            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2

            txt_loc = (xmin, ymax + 20)
            cv2.putText(img,
                        text="Gender Acc: " + str(info["gender"][i])[:4],
                        org=txt_loc,
                        fontFace=font,
                        fontScale=font_scale,
                        color=color_bbox,
                        thickness=thickness)
            txt_loc = (xmin, ymax + 40)
            cv2.putText(img,
                        text="Age: " + str(info["age_mean"][i])[:4] + " +/- " + str(info["age_mae"][i])[:4],
                        org=txt_loc,
                        fontFace=font,
                        fontScale=font_scale,
                        color=color_bbox,
                        thickness=thickness)
            txt_loc = (xmin, ymax + 60)
            cv2.putText(img,
                        text="Age STD: " + str(info["age_std"][i])[:4],
                        org=txt_loc,
                        fontFace=font,
                        fontScale=font_scale,
                        color=color_bbox,
                        thickness=thickness)
            txt_loc = (xmin, ymax + 80)
            cv2.putText(img,
                        text="GT: " + str(info["gt_gender"][i]) + "/" + str(info["gt_age"][i]),
                        org=txt_loc,
                        fontFace=font,
                        fontScale=font_scale,
                        color=[0, 0, 255],
                        thickness=thickness)


def draw_act(img, boxes, act_s, act_lbs, color_bbox=(0, 225, 0), thickness=2):
    act_lbs = np.asarray(act_lbs)
    info = np.zeros(5, dtype=int)
    for bbox, act in zip(boxes, act_s):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        text = str(act)
        # GET INFO
        if text in act_lbs:
            idx = np.where(act_lbs == text)
            info[idx] += 1
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, thickness=thickness)
        txt_loc = (xmin, ymin - 40)
        cv2.putText(img,
                    text=text,
                    org=txt_loc,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color_bbox,
                    thickness=thickness)
    return img, info


def draw_act_trackers(img, tracks, act_lbs, num_vote=10, color=(0, 225, 0), thickness=2, draw_id=False):
    act_lbs = np.asarray(act_lbs)
    info = np.zeros(5, dtype=int)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        xmin, ymin, xmax, ymax = [int(x) for x in track.to_tlbr()]

        # TRACK ID
        if draw_id:
            color = create_unique_color(track.track_id)
            label = str(track.track_id)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img,
                          (xmin, ymin),
                          (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                          color, -1)
            cv2.putText(img, label,
                        (xmin + 5, ymin + 5 + text_size[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        # VOTING
        if len(track.actions) > 0:
            act_arr = np.asarray(track.actions[-num_vote:]) if len(track.actions) > num_vote else np.asarray(track.actions)
            act, fr = np.unique(act_arr, return_counts=True)
            id_max = np.argmax(fr)
            text = str(act[id_max])
        else:
            text = " "

        # GET INFO
        if text in act_lbs:
            idx = np.where(act_lbs == text)
            info[idx] += 1
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=thickness)
        txt_loc = (xmin, ymin - 40)
        cv2.putText(img,
                    text=text,
                    org=txt_loc,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color,
                    thickness=thickness)
    return img, info


def draw_bbox_option(img, bbox, draw_type="rect", color_bbox=(204, 102, 0)):
    """
    :param img: images need ro draw box
    :param bbox: bounding box, format [xmin, ymin, xmax, ymax]
    :param draw_type: rectangle or corner
    :param color_bbox: color of bounding box
    :return: bounding box
    """
    thickness = 2
    xmin, ymin, xmax, ymax = np.asarray(bbox, dtype=int)
    width, height = xmax - xmin, ymax - ymin
    ratio = 0.25
    # draw
    if draw_type == "rect":
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, thickness=thickness)
        return img
    else:
        # Upper left corner
        pt1 = (xmin, ymin)
        pt2 = (int(xmin + ratio * width), ymin)
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        pt1 = (xmin, ymin)
        pt2 = (xmin, int(ymin + ratio * height))
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        # Upper right corner
        pt1 = (xmax, ymin)
        pt2 = (xmax, int(ymin + ratio * height))
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        pt1 = (xmax, ymin)
        pt2 = (int(xmax - ratio * width), ymin)
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        # Lower left corner
        pt1 = (xmin, ymax)
        pt2 = (xmin, int(ymax - ratio * height))
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        pt1 = (xmin, ymax)
        pt2 = (int(xmin + ratio * width), ymax)
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        # Lower right corner
        pt1 = (xmax, ymax)
        pt2 = (xmax, int(ymax - ratio * height))
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        pt1 = (xmax, ymax)
        pt2 = (int(xmax - ratio * width), ymax)
        cv2.line(img, pt1, pt2, color=color_bbox, thickness=thickness)

        return img


def draw_bbox_with_info(image, info, option="corner", color_bbox=(0, 255, 255), color_text=(0, 255, 255)):
    """
    :param image: images need ro draw box
    :param info: format type: dictionary, include {"bbox", "age", "gender", "expression"}
    :param option: rectangle or corner, "rect", "corner"
    :param color_bbox: color of bounding box
    :param color_text: color of text
    :return: image has been drawn box with information
    """
    for i, bbox in enumerate(info["bbox"]):
        # init
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        # # Age
        age = str("Age: {}".format(int(info["age"][i])))
        # # Gender
        gender = str("Gen: {}".format(info["gender"][i]))
        # # # Expression
        # expression = str("Exp: {}".format(info["expression"]))

        # draw bbox
        draw_bbox_option(image, bbox, draw_type=option, color_bbox=color_bbox)

        # draw info
        font_scale = 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        # Age
        txt_loc = (xmin, ymax + 20)
        cv2.putText(image,
                    text=age,
                    org=txt_loc,
                    fontFace=font,
                    fontScale=font_scale,
                    color=color_text,
                    thickness=thickness)
        # Gender
        txt_loc = (xmin, ymax + 40)
        cv2.putText(image,
                    text=gender,
                    org=txt_loc,
                    fontFace=font,
                    fontScale=font_scale,
                    color=color_text,
                    thickness=thickness)
        # # # Expression
        # txt_loc = (xmin, ymax + 60)
        # cv2.putText(img,
        #             text=expression,
        #             org=txt_loc,
        #             fontFace=font,
        #             fontScale=font_scale,
        #             color=color_text,
        #             thickness=thickness)


def draw_bbox_with_text(image, bbox, text, color=(0, 0, 255)):
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    color = color
    # draw bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    # track id
    label = text
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.rectangle(image,
                  (xmin, ymin - 10 - text_size[0][1]),
                  (xmin + 10 + text_size[0][0], ymin),
                  color, -1)
    cv2.putText(image, label,
                (xmin + 5, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)


def draw_bbox_with_id(image, bbox, id):
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    color = (255, 0, 0)
    # draw bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    # track id
    label = "ID: " + str(id)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image,
                  (xmin, ymin),
                  (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                  color, -1)
    cv2.putText(image, label,
                (xmin + 5, ymin + 5 + text_size[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)


def draw_bbox_without_id(image, bbox):
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    color = (255, 0, 0)
    # draw bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)


def draw_trackers(image, tracks, accumulation_gender=True, accumulation_age=True, show_both=False, draw_id=True):
    '''
    show_both: True will show accumulate_current
    '''
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        xmin, ymin, xmax, ymax = [int(x) for x in track.to_tlbr()]
        color = create_unique_color(track.track_id)

        # draw bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # track id
        if draw_id:
            label = str(track.track_id)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image,
                          (xmin, ymin),
                          (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                          color, -1)
            cv2.putText(image, label,
                        (xmin + 5, ymin + 5 + text_size[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        # draw info
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        # Gender
        gender=[]
        if len(track.genders) > 0:
            if accumulation_gender:
                # return accumulate gender
                gender.append("F") if np.mean(track.genders) <= 0.6 else gender.append("M")
                if show_both:
                    # gender = "F" if np.mean(track.genders) <= 0.6 else "M"
                    gender.append("F") if track.genders[-1] < 0.5 else gender.append("M")
                
            else:
                gender = "F" if track.genders[-1] < 0.5 else "M"

        else:
            gender = "N/A"
        # Age
        age = []
        if len(track.ages) > 0:
            if accumulation_age:
                # age = str(int(np.mean(track.ages)))
                age.append(str(int(np.mean(track.ages))))
                if  show_both:
                    age.append(str(int(track.ages[-1])))
            else:
                age.append(str(int(track.ages[-1])))
        else:
            age.append("N/A") 

        # GENDER AGE
        gen_age = "_".join(gender) + "_".join(age)
        text_size = cv2.getTextSize(gen_age, font, font_scale, thickness)
        txt_loc = (xmin, ymax + 10 + text_size[0][1])
        # cv2.putText(image,
        #             text=gen_age,
        #             org=txt_loc,
        #             fontFace=font,
        #             fontScale=font_scale,
        #             color=color,
        #             thickness=thickness)

        # EXPR
        text_size = cv2.getTextSize(track.expr, font, font_scale, thickness)
        # txt_loc = (xmin, ymax + 30 + 2*text_size[0][1])
        cv2.putText(image,
                    text=track.expr,
                    org=txt_loc,
                    fontFace=font,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness)


def draw_trackers_info(image, tracks, list_expr, info,
                       accumulation_gender=True, accumulation_age=True, draw_id=True):
    # info = np.zeros(8, dtype=int)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        xmin, ymin, xmax, ymax = [int(x) for x in track.to_tlbr()]
        color = create_unique_color(track.track_id)

        # draw bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # track id
        if draw_id:
            label = str(track.track_id)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image,
                          (xmin, ymin),
                          (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                          color, -1)
            cv2.putText(image, label,
                        (xmin + 5, ymin + 5 + text_size[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        # draw info
        font_scale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        # Gender
        if len(track.genders) > 0:
            if accumulation_gender:
                # return accumulate gender at last 200 frames of thisID
                _gen = np.mean(track.genders[-200:]) if len(track.genders) > 200 else np.mean(track.genders)
            else:
                _gen = track.genders[-1]

            gender = "F" if _gen < 0.6 else "M"
            # GET INFO FOR STATISTICAL
            idx_g = 7 if _gen < 0.6 else 6
            info[idx_g] += 1
        else:
            gender = "N/A"
        # Expr
        if track.expr in list_expr:
            idx = np.where(list_expr == str(track.expr))
            info[idx] += 1
        # Age
        if len(track.ages) > 0:
            if accumulation_age:
                # return accumulate age of last 200 images of this ID, or last all images if <200
                # age_acc = str(int(np.mean(track.ages[-200:]))) if len(track.ages) > 200 else str(int(np.mean(track.ages)))
                # Modify to show every 5s
                i = len(track.ages) % 5
                if i !=0:
                    # bỏ đi những cái cuối
                    if len(track.ages) > 200:
                        age = str(int(np.mean(track.ages[(-200-i): (-i)])))
                    else:
                        if len(track.ages) > 5:
                            age =  str(int(np.mean(track.ages[:-i])))
                        else:
                            age =  str(int(np.mean(track.ages)))
                else:
                    #update cai cuối
                    age= str(int(np.mean(track.ages[-200:]))) if len(track.ages) > 200 else str(int(np.mean(track.ages)))
                    print('Update')

            else:
                age = str(int(track.ages[-1]))
        else:
            age = "N/A"
        
        # GENDER AGE
        gen_age = gender + age
        text_size = cv2.getTextSize(gen_age, font, font_scale, thickness)
        txt_loc = (xmin, ymax + 10 + text_size[0][1])
        cv2.putText(image,
                    text=gen_age,
                    org=txt_loc,
                    fontFace=font,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness)

        # EXPR
        text_size = cv2.getTextSize(track.expr, font, font_scale, thickness)
        txt_loc = (xmin, ymax + 30 + 2*text_size[0][1])
        cv2.putText(image,
                    text=track.expr,
                    org=txt_loc,
                    fontFace=font,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness)
    # return info


def draw_trackers_without_info(image, tracks):
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        xmin, ymin, xmax, ymax = [int(x) for x in track.to_tlbr()]
        color = create_unique_color(track.track_id)

        # draw bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # track id
        label = str(track.track_id)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image,
                      (xmin, ymin),
                      (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                      color, -1)
        cv2.putText(image, label,
                    (xmin + 5, ymin + 5 + text_size[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)


def draw_bbox_with_gender_info(image, info, option="corner", color_bbox=(0, 255, 255), color_text=(0, 255, 255)):
    """
    :param image: images need ro draw box
    :param info: format type: dictionary, include {"bbox", "age", "gender", "expression"}
    :param option: rectangle or corner, "rect", "corner"
    :param color_bbox: color of bounding box
    :param color_text: color of text
    :return: image has been drawn box with information
    """
    # init
    xmin, ymin, xmax, ymax = np.asarray(info["bbox"], dtype=int)

    if info["gender"] == "Male":
        color = (204, 102, 0)
    else:
        color = (0, 255, 255)
    # draw bbox
    img = draw_bbox_option(image, info["bbox"], draw_type=option, color_bbox=color)

    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # # draw info
    # font_scale = 0.6
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # thickness = 2
    # # # Age
    txt_loc = (xmin, ymax + 20)
    cv2.putText(img,
                text=info["age"],
                org=txt_loc,
                fontFace=font,
                fontScale=font_scale,
                color=color_text,
                thickness=thickness)
    # # # Gender
    # txt_loc = (xmin, ymax + 40)
    # cv2.putText(img,
    #             text=gender,
    #             org=txt_loc,
    #             fontFace=font,
    #             fontScale=font_scale,
    #             color=color_text,
    #             thickness=thickness)
    # # Expression
    txt_loc = (xmin, ymax + 40)
    cv2.putText(img,
                text=info["expression"],
                org=txt_loc,
                fontFace=font,
                fontScale=font_scale,
                color=color_text,
                thickness=thickness)


def draw_age(img, boxes, ages, color_bbox=(204, 102, 0)):
    for idx, bbox in enumerate(boxes):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, 2)
        txt_loc = (xmin, ymax + 40)
        cv2.putText(img,
                    text=str(int(ages[idx])),
                    org=txt_loc,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color_bbox,
                    thickness=2)
    return img


def crop_faces(boxes, input_img, ratio=0.2, size=(112, 112)):
    faces = []
    boxes = np.asarray(boxes)
    if np.ndim(boxes) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    img_h, img_w, _ = np.shape(input_img)
    if len(boxes) > 0:
        for i, bb in enumerate(boxes):
            xmin, ymin, xmax, ymax = bb[:4]
            w = xmax - xmin
            h = ymax - ymin
            xw1 = max(int(xmin - ratio * w), 0)
            yw1 = max(int(ymin - ratio * h), 0)
            xw2 = min(int(xmax + ratio * w), img_w - 1)
            yw2 = min(int(ymax + ratio * h), img_h - 1)
            crop = input_img[yw1:yw2, xw1:xw2]
            crop = cv2.resize(crop, size)
            faces.append(crop)
    return faces


def draw_head_pose(img, bboxes, head_poses, size=50):
    for i, bbox in enumerate(bboxes):
        yaw, pitch, roll = head_poses[i][0], head_poses[i][1], head_poses[i][2]

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[:4]
        tdx = (bbox_x2 + bbox_x1) / 2
        tdy = (bbox_y2 + bbox_y1) / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    """crop from center """
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


# def preprocess_face(faces):
#     faces_160_160 = []
#     for face in faces:
#         # face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         face_rgb = cv2.resize(face_rgb, (160, 160))
#         prewhitened = prewhiten(face_rgb)
#         # cropped = crop(prewhitened, False, 160)
#         cropped = prewhitened
#         faces_160_160.append(cropped)
#         # cv2.imshow("Test", cropped)
#         # cv2.waitKey()
#     return np.array(faces_160_160)


def preprocess_face(faces):
    """
    whitened and resize
    """
    faces_160_160 = []
    for face in faces:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        prewhitened = prewhiten(face_rgb)
        cropped = crop(prewhitened, False, 160)
        faces_160_160.append(cropped)
    return np.array(faces_160_160)


def preprocess_face_facereid(faces):
    faces_160_160 = []
    for face in faces:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face_rgb, (182, 182))
        prewhitened = prewhiten(face)
        cropped = crop(prewhitened, False, 160)
        faces_160_160.append(cropped)
    return np.array(faces_160_160)


def crop_bboxes(image, bboxes):
    imgs = []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(int)
        img = image[ymin:ymax, xmin:xmax]
        imgs.append(img)
    return imgs
