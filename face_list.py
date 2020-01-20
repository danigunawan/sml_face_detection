import numpy as np
import cv2

class Face_List():

    def __init__(self):

        # Face container
        self.face_container = []

        # Face's feature container
        self.feature_container = []

        # Duration container
        self.duration_container = []

        # Emotion container
        self.emotion_container = []

        # Number of face
        self.num_of_face = 0

    # Adding face
    def add_new_face(self, face, feature, emotion, frame):
        self.face_container.append(face)
        self.feature_container.append(feature)
        self.duration_container.append([frame])
        self.emotion_container.append([emotion])
        self.num_of_face += 1

    # Save
    def save_list(self, count):
        # count = 0
        # for img in self.face_container:
        #     cv2.imwrite('img'+str(count)+'.jpg', img)

        np.save('database/face_container_' + str(count), self.face_container, fix_imports=True)
        np.save('database/duration_container_' + str(count), self.duration_container, fix_imports=True)
        np.save('database/emotion_container_' + str(count), self.emotion_container, fix_imports=True)
        np.save('database/num_of_face_' + str(count), self.num_of_face, fix_imports=True)


