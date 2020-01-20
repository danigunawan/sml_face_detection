"""
aioz.aiar.truongle - Dec 30, 2019
build database for face search
"""
import cv2
import os
def main():
    cam = "cam02"
    name = "191215_016.mp4"
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
                cv2.imwrite('first_frame_' + name.replace('mp4', 'jpg'), frame)
                break
        cap.release()


if __name__ == '__main__':
    main()



