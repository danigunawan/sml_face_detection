"""
aioz.aiar.truongle
gender wrapper
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models, layers, backend
# TTA for GENDER CLASSIFICATION
import imgaug as ia
from tensorflow.python.keras.utils import Sequence
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
    ],
    random_order=True)
# test_time augmentation number
TTA_NUM = 5


class My_Generator(Sequence):

    def __init__(self, images,
                 batch_size):

        self.images = images
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.test_generate(batch_x)

    def test_generate(self, batch_x):
        batch_images = []
        for img in batch_x:
            for i in range(TTA_NUM):
                img_aug = seq.augment_image(img)
                batch_images.append(img_aug)
        batch_images = np.array(batch_images)
        return batch_images


class GenderDetectionPb:
    """
    Load pb file
    """
    def __init__(self, config):
        # LOAD GRAPH
        self.graph_path = config.gender_pb_path
        graph = self._load_graph()
        self.sess = tf.Session(graph=graph)
        self.tf_inputs = graph.get_tensor_by_name("import/input_1:0")
        self.tf_outputs = graph.get_tensor_by_name('import/lambda/Sigmoid:0')

    def _load_graph(self):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(self.graph_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def _process_with_tta(self, faces):
        faces = np.squeeze(My_Generator(faces, batch_size=len(faces)))
        gender_outs = self.sess.run(self.tf_outputs, feed_dict={self.tf_inputs: faces})
        gender_outs = np.reshape(gender_outs, (-1, TTA_NUM))
        gender_outs = np.mean(gender_outs, axis=1)  # [1, num_faces]

        # HARD : <0.5 = 0 , >0.5 =1
        # mask = np.rint(gender_outs)

        # SOFT & FILTER : only choose gender (0:0,4) or (0.6,1)
        # mask = np.logical_or((gender_outs < 0.4), (gender_outs > 0.6))
        # gender_outs[~mask] = None
        # genders = np.where(np.isnan(gender_outs), None, gender_outs)

        # Tuning|Trick
        mask = gender_outs <= 0.6
        gender_outs[mask] = gender_outs[mask] / 2.0
        genders = np.rint(gender_outs)
        return genders

    def _process_without_tta(self, faces):
        gender_outs = self.sess.run(self.tf_outputs, feed_dict={self.tf_inputs: faces})
        # genders = [gen[0] if gen < 0.4 or gen > 0.6 else None for gen in gender_outs]
        # HARD
        genders = [0 if gen < 0.6 else 1 for gen in gender_outs]
        return genders

    def process_prediction(self, faces, use_tta=False):
        assert np.ndim(faces) == 4 or np.ndim(faces) == 3
        if np.ndim(faces) == 3:
            faces = np.expand_dims(faces, axis=0)
        else:
            faces = faces
        if use_tta:
            genders = self._process_with_tta(faces)
        else:
            genders = self._process_without_tta(faces)

        return genders


class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


class GenderDetectionH5:
    """
    Load h5 file
    """
    def __init__(self, config):
        def swish(x):
            return tf.nn.swish(x)
        self.gender_net = models.load_model(config.gender_h5_path,
                                            custom_objects={'swish': swish, 'tf': tf, 'FixedDropout': FixedDropout},
                                            compile=False)

    def process_prediction(self, faces):
        gender_outs = self.gender_net.predict(faces)
        # HARD
        # genders = [1 if gender_out > 0.5 else 0 for gender_out in gender_outs]
        # SOFT & FILTER
        genders = [gen[0] if gen < 0.4 or gen > 0.6 else None for gen in gender_outs]
        # return genders
        return list(gender_outs)

    def process_aug_prediction(self, faces):
        """
        test time augmentation
        """
        gender_outs = self.gender_net.predict_generator(My_Generator(faces, batch_size=len(faces)))
        gender_outs = np.reshape(gender_outs, (-1, TTA_NUM))
        gender_outs = np.mean(gender_outs, axis=1)  # [1, num_faces]

        # HARD : <0.5 = 0 , >0.5 =1
        # mask = np.rint(gender_outs)

        # SOFT & FILTER : only choose gender (0:0,4) or (0.6,1)
        # mask = np.logical_or((gender_outs < 0.4), (gender_outs > 0.6))
        # gender_outs[~mask] = None
        # genders = np.where(np.isnan(gender_outs), None, gender_outs)

        # Tunning|Trick.  Divide probability of female
        mask = gender_outs <= 0.5  # TODO tune parameter. [0.5 to 0.6]
        gender_outs[mask] = gender_outs[mask]/2.0
        genders = gender_outs

        return genders
