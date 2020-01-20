from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mxnet as mx
from sklearn import preprocessing


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


class InsightFace:
    def __init__(self, config, gpu_id=0):
        self.config = config
        self.im_size = config.insightface_im_size
        self.model_str = config.insightface_model
        self.ctx = mx.gpu(gpu_id)
        self.model = None
        self._get_model()

    def _get_model(self, layer='fc1'):
        _vec = self.model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('[INFO] loading model insight face ', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, self.im_size[0], self.im_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def get_feature(self, face):
        """face RGB mode"""
        assert np.ndim(face) == 4 or np.ndim(face) == 3
        if np.ndim(face) == 3:
            input_blob = np.expand_dims(face, axis=0)
        else:
            input_blob = face
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding)
        return embedding
