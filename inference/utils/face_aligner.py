"""
aioz.aiar.truongle - Oct 08, 2019
face alignment
"""
import face_alignment
import cv2
import dlib
import numpy as np


class FaceAligner:
    def __init__(self, config, mode="dlib", equalize_hist=False, sharpened=False,
                 landmark_points='68', chipSize=300, margin=0.2):
        """mode = 'dlib' or 'face_al'
        """
        self.landmark_points = landmark_points
        self.equalize_hist = equalize_hist
        self.sharpened = sharpened
        self.margin = margin
        self.use_dlib = True if mode == "dlib" else False
        if landmark_points == '68':
            aligner_path = config.aligner
        else:
            aligner_path = config.aligner5

        if self.use_dlib:
            self.face_regressor = dlib.shape_predictor(aligner_path)
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                   flip_input=False, face_detector='dlib',
                                                   device="cuda")
        # Initialize the chip resolution
        self.chipSize = chipSize
        # define 4 corners of image to use perspective mapping
        self.chipCorners = np.float32([[0, 0],
                                       [self.chipSize, 0],
                                       [0, self.chipSize],
                                       [self.chipSize, self.chipSize]])

    @staticmethod
    def _equalize_hist_face(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        return face

    @staticmethod
    def _sharpened_face(face):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(face, -1, kernel)
        return sharpened

    @staticmethod
    def _crop_img_by_rect(img, rect, margin=0.2):
        x1, y1, x2, y2 = rect[:4]
        # size of face
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # add margin
        full_crop_x1 = x1 - int(w * margin)
        full_crop_y1 = y1 - int(h * margin)
        full_crop_x2 = x2 + int(w * margin)
        full_crop_y2 = y2 + int(h * margin)

        # size of face with margin
        new_size_w = full_crop_x2 - full_crop_x1 + 1
        new_size_h = full_crop_y2 - full_crop_y1 + 1

        # ensure that the region cropped from the original image with margin
        # doesn't go beyond the image size
        crop_x1 = max(full_crop_x1, 0)
        crop_y1 = max(full_crop_y1, 0)
        crop_x2 = min(full_crop_x2, img.shape[1] - 1)
        crop_y2 = min(full_crop_y2, img.shape[0] - 1)
        # size of the actual region being cropped from the original image
        crop_size_w = crop_x2 - crop_x1 + 1
        crop_size_h = crop_y2 - crop_y1 + 1

        # coordinates of region taken out of the original image in the new image
        new_location_x1 = crop_x1 - full_crop_x1
        new_location_y1 = crop_y1 - full_crop_y1
        new_location_x2 = crop_x1 - full_crop_x1 + crop_size_w - 1
        new_location_y2 = crop_y1 - full_crop_y1 + crop_size_h - 1

        new_img = np.random.randint(256, size=(new_size_h, new_size_w, img.shape[2])).astype('uint8')

        new_img[new_location_y1: new_location_y2 + 1, new_location_x1: new_location_x2 + 1, :] = \
            img[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1, :]

        # if margin goes beyond the size of the image, repeat last row of pixels
        if new_location_y1 > 0:
            new_img[0:new_location_y1, :, :] = np.tile(new_img[new_location_y1, :, :], (new_location_y1, 1, 1))

        if new_location_y2 < new_size_h - 1:
            new_img[new_location_y2 + 1:new_size_h, :, :] = np.tile(new_img[new_location_y2:new_location_y2 + 1, :, :],
                                                                    (new_size_h - new_location_y2 - 1, 1, 1))
        if new_location_x1 > 0:
            new_img[:, 0:new_location_x1, :] = np.tile(new_img[:, new_location_x1:new_location_x1 + 1, :],
                                                       (1, new_location_x1, 1))
        if new_location_x2 < new_size_w - 1:
            new_img[:, new_location_x2 + 1:new_size_w, :] = np.tile(new_img[:, new_location_x2:new_location_x2 + 1, :],
                                                                    (1, new_size_w - new_location_x2 - 1, 1))
        return new_img

    def _align68(self, faces):
        aligned_faces = []
        for im in faces:
            # PRE-PROCESS
            image = cv2.resize(im, (self.chipSize, self.chipSize))
            bb = np.array([[0, 0, image.shape[0], image.shape[1]]])
            if self.use_dlib:
                # dlib_img = image[..., ::-1].astype(np.uint8)
                dlib_img = image
                xmin, ymin, xmax, ymax = bb[0].astype(int)
                dlib_box = dlib.rectangle(left=xmin, top=ymin, right=xmax, bottom=ymax)
                pts = self.face_regressor(dlib_img, dlib_box).parts()
            else:
                pts = self.fa.get_landmarks_from_image(image, detected_faces=bb)[0]
            if pts is not None:
                if self.use_dlib:
                    lmk = np.array([[pt.x, pt.y] for pt in pts])
                else:
                    lmk = np.array([[pt[0], pt[1]] for pt in pts])
                # # DEBUG
                # for p in lmk:
                #     x, y = p
                #     cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                # Compute the Anchor Landmarks
                # This ensures the eyes and chin will not move within the chip
                rightEyeMean = np.mean(lmk[36:42], axis=0)
                leftEyeMean = np.mean(lmk[42:48], axis=0)
                middleEye = (rightEyeMean + leftEyeMean) * 0.5
                chin = lmk[8]
                # Compute the chip center and up/side vectors
                mean = ((middleEye * 3) + chin) * 0.25
                rightVector = (leftEyeMean - rightEyeMean)
                upVector = (chin - middleEye)

                # Divide by the length ratio to ensure a square aspect ratio
                rightVector /= np.linalg.norm(rightVector) / np.linalg.norm(upVector)

                # Compute the corners of the facial chip
                imageCorners = np.float32([(mean + (-rightVector - upVector))[:2],
                                           (mean + (rightVector - upVector))[:2],
                                           (mean + (-rightVector + upVector))[:2],
                                           (mean + (rightVector + upVector))[:2]])
                # FILTER
                imageCorners = np.where(imageCorners > 0, imageCorners, 0)
                imageCorners = np.where(imageCorners < self.chipSize, imageCorners, self.chipSize)
                # Compute the Perspective Homography and Extract the chip from the image
                chipMatrix = cv2.getPerspectiveTransform(imageCorners, self.chipCorners)
                face = cv2.warpPerspective(image, chipMatrix, (self.chipSize, self.chipSize))
            else:
                face = image

            # POST PROCESSING
            if self.sharpened:
                face = self._sharpened_face(face)
            if self.equalize_hist:
                face = self._equalize_hist_face(face)
            aligned_faces.append(face)
        return aligned_faces

    def align(self, faces):
        return self._align68(faces)
