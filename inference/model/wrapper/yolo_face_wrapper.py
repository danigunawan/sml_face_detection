"""
aioz.aiar.truongle - 14 Dec, 2019
yolo face detection
ref: https://github.com/sthanhng/yoloface
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def letterbox_image(image, size):
    """Resize image with unchanged aspect ratio using padding"""

    img_width, img_height = image.size
    w, h = size
    scale = min(w / img_width, h / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


class YoloFace:
    def __init__(self, config, memory_fraction=0.3):
        # config gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = False

        self.graph_fp = config.yolo_face_graph
        self.model_image_size = (416, 416)
        self.threshold = config.yolo_face_thrs
        self.session = None

        self._load_graph()
        self._init_predictor()

    def _load_graph(self):
        print('[INFO] Load graph at {} ... '.format(self.graph_fp))
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # tf.get_default_graph().finalize()

    def _init_predictor(self):
        # print('[INFO] Init predictor ...')
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=self.config)
            self.image_tensor = self.graph.get_tensor_by_name('input_1:0')
            self.image_shape_tensor = self.graph.get_tensor_by_name('input_shape_tensor:0')
            self.boxes_tensor = self.graph.get_tensor_by_name('concat_11:0')
            self.score_tensor = self.graph.get_tensor_by_name('concat_12:0')
            self.labels_tensor = self.graph.get_tensor_by_name('concat_13:0')

    def process_detection(self, image, filter_size=None):
        im_h, im_w, _ = image.shape
        image = Image.fromarray(image)
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # add batch dimension
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.session.run(
            [self.boxes_tensor, self.score_tensor, self.labels_tensor],
            feed_dict={
                self.image_tensor: image_data,
                self.image_shape_tensor: [image.size[1], image.size[0]],
            })

        # filter
        indices_ps = np.where(out_classes == 0)
        boxes_ps = out_boxes[indices_ps]
        scores_ps = out_scores[indices_ps]
        indices = np.where(scores_ps > self.threshold)
        _boxes = boxes_ps[indices]

        # convert boxes
        boxes = np.zeros(_boxes.shape)
        boxes[:, 0] = _boxes[:, 1]
        boxes[:, 1] = _boxes[:, 0]
        boxes[:, 2] = _boxes[:, 3]
        boxes[:, 3] = _boxes[:, 2]

        # check boxes out off size
        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], im_w)
        boxes[:, 3] = np.minimum(boxes[:, 3], im_h)

        if filter_size is not None:
            if len(boxes) > 0:
                boxes_mask = ((boxes[:, 3] - boxes[:, 1]) > filter_size) * \
                             ((boxes[:, 2] - boxes[:, 0]) > filter_size)
                boxes = boxes[boxes_mask]

        return boxes.astype(int)

