"""
config for all modules
"""


class Config:
    def __init__(self):
        # FACE ALIGNER
        self.aligner = 'inference/utils/face_aligner/shape_predictor_68_face_landmarks.dat'
        self.aligner5 = 'inference/utils/face_aligner/shape_predictor_5_face_landmarks.dat'
        self.aligner_targets = 'inference/utils/face_aligner/targets_symm.txt'

        # INSIGHT FACE
        self.insightface_model = "models/insightface_model, 0"
        self.insightface_im_size = (112, 112)

        # YOLO FACE
        self.yolo_face_graph = 'models/2019_12_14_yolo_face.pb'
        self.yolo_face_thrs = 0.01

        # GENDER DETECTION
        self.gender_h5_path = 'models/2019_09_13_gender_EfficientNetB4_model.h5'
        self.gender_pb_path = 'models/2019_09_13_gender_EfficientNetB4_model.pb'

        # EXPRESSION
        self.expr_h5_path = 'models/2019_10_10_emo_FER_effb1_fcl_affineScale_illumination_model.h5'
        self.expr_pb_path = 'models/2019_10_10_emo_FER_effb1_fcl_affineScale_illumination_model.pb'
        # self.exprs = ["Neu", "Hap", "Sur", "Sad", "Ang", "Fear"]
        self.exprs = ["Neu", "Hap", "Neu", "Sad", "Neu", "Neu"]

        # HEAD POSE FSAnet
        self.fsanet_graph_path = 'models/2019_08_15_HeadPose_FSA.pb'
