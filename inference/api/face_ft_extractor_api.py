"""
aioz.aiar.truongle - Dec 31, 2019
face feature extractor
"""
from ..utils.face_aligner import FaceAligner
from ..models.wrapper.insightface_face_ft_extractor_wrapper import InsightFace


class FaceFtExtractorApi:
    def __init__(self, config, align_mode='non'):
        """align_mode include 3 mode: face_al, dlib, non
        """
        # INIT PARAMETER
        self.align_mode = align_mode
        self.face_size = 112
        self.config = config
        self._init_all_model()

    def _init_all_model(self):
        # FACE ALIGNER: improve speed with chip_size=112 because model the same input size
        self.face_aligner = FaceAligner(self.config, mode=self.align_mode,
                                        equalize_hist=False, sharpened=False,
                                        landmark_points='68', chipSize=self.face_size)
        # FACE FEATURE EXTRACTOR
        self.face_ft_extractor = InsightFace(config=self.config)

    def proceed(self, faces):
        faces_align = self.face_aligner.align(faces)
        face_fts = self.face_ft_extractor.get_feature(faces_align)

        return face_fts
