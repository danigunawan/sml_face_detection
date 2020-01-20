import tensorflow as tf
import numpy as np
import os


class HeadPoseDetection:
    def __init__(self, config):
        self.config = config
        # Load head pose detection graph
        self.graph_path = self.config.fsanet_graph_path
        graph = self._load_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=sess_config)
        self.tf_input = graph.get_tensor_by_name('import/input_27:0')
        self.tf_output = graph.get_tensor_by_name('import/average_1/truediv:0')

    def _load_graph(self):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(self.graph_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def process_prediction(self, images, threshold=None):
        """
        image size: 64*64
        """
        head_pose_mask = None
        outputs = self.sess.run([self.tf_output], feed_dict={self.tf_input: images})
        head_poses = np.asarray(outputs)[0]
        # GET HEAD POSE MASK
        if threshold is not None:
            head_pose_mask = np.sum(np.abs(head_poses) > threshold, axis=1) == 0
        return head_poses, head_pose_mask
