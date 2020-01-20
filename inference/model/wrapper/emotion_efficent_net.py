"""
aioz.aiar.truongle
emotion wrapper
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models, layers, backend


class ExpressionDetectionPb:
    """
    Load pb file
    """
    def __init__(self, config):
        self.ths_hold = 0.4
        self.expressions = config.exprs
        # LOAD GRAPH
        self.graph_path = config.expr_pb_path
        graph = self._load_graph()
        self.sess = tf.Session(graph=graph)
        self.tf_inputs = graph.get_tensor_by_name("import/input_1:0")
        self.tf_outputs = graph.get_tensor_by_name('import/dense_1/Softmax:0')

    def _load_graph(self):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(self.graph_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def process_prediction(self, faces):
        assert np.ndim(faces) == 4 or np.ndim(faces) == 3
        if np.ndim(faces) == 3:
            faces = np.expand_dims(faces, axis=0)
        else:
            faces = faces
        expr_outs = self.sess.run(self.tf_outputs, feed_dict={self.tf_inputs: faces})
        expr_idx = np.argmax(expr_outs, 1)
        # filter
        emo_max = np.max(expr_outs, axis=1)
        indices = np.where(emo_max < self.ths_hold)[0]
        expr_idx[indices] = -1
        exprs = [self.expressions[x] if x != -1 else " " for x in expr_idx]
        return exprs


class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


class ExpressionDetectionH5:
    """
    Load h5 file
    """
    def __init__(self, config):
        self.ths_hold = 0.4
        self.expressions = config.exprs

        def swish(x):
            return tf.nn.swish(x)

        self.expr_net = models.load_model(config.expr_h5_path,
                                          custom_objects={'swish': swish, 'FixedDropout': FixedDropout},
                                          compile=False)

    def process_prediction(self, faces):
        expr_outs = self.expr_net.predict(faces)
        expr_idx = np.argmax(expr_outs, 1)
        # filter
        emo_max = np.max(expr_outs, axis=1)
        indices = np.where(emo_max < self.ths_hold)[0]
        expr_idx[indices] = -1
        exprs = [self.expressions[x] if x != -1 else " " for x in expr_idx]
        return exprs

