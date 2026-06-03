# coding: utf-8
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.gradient import numerical_gradient
from common.layers import Affine, Convolution, Dropout, Pooling, Relu, SoftmaxWithLoss


class SimpleConvNet:
    """NumPy CNN based on the book's layer implementations."""

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param=None,
        hidden_size=256,
        output_size=10,
        weight_init_std="he",
        dropout_ratio=0.35,
        weight_decay_lambda=1e-4,
    ):
        if conv_param is None:
            conv_param = {
                "filter_nums": (32, 32, 64, 64),
                "filter_size": 3,
                "pad": 1,
                "stride": 1,
            }

        self.weight_decay_lambda = weight_decay_lambda
        filter_nums = conv_param.get("filter_nums", (32, 32, 64, 64))
        filter_size = conv_param.get("filter_size", 3)
        filter_pad = conv_param.get("pad", 1)
        filter_stride = conv_param.get("stride", 1)

        def init_scale(fan_in):
            if str(weight_init_std).lower() in ("he", "relu"):
                return np.sqrt(2.0 / fan_in)
            return float(weight_init_std)

        self.params = {}
        in_channels = input_dim[0]
        for idx, filter_num in enumerate(filter_nums, start=1):
            fan_in = in_channels * filter_size * filter_size
            self.params[f"W{idx}"] = init_scale(fan_in) * np.random.randn(
                filter_num, in_channels, filter_size, filter_size
            )
            self.params[f"b{idx}"] = np.zeros(filter_num)
            in_channels = filter_num

        pooled_h = input_dim[1] // 4
        pooled_w = input_dim[2] // 4
        affine_input_size = filter_nums[-1] * pooled_h * pooled_w
        self.params["W5"] = init_scale(affine_input_size) * np.random.randn(
            affine_input_size, hidden_size
        )
        self.params["b5"] = np.zeros(hidden_size)
        self.params["W6"] = init_scale(hidden_size) * np.random.randn(
            hidden_size, output_size
        )
        self.params["b6"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"], self.params["b1"], filter_stride, filter_pad
        )
        self.layers["Relu1"] = Relu()
        self.layers["Conv2"] = Convolution(
            self.params["W2"], self.params["b2"], filter_stride, filter_pad
        )
        self.layers["Relu2"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Conv3"] = Convolution(
            self.params["W3"], self.params["b3"], filter_stride, filter_pad
        )
        self.layers["Relu3"] = Relu()
        self.layers["Conv4"] = Convolution(
            self.params["W4"], self.params["b4"], filter_stride, filter_pad
        )
        self.layers["Relu4"] = Relu()
        self.layers["Pool2"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W5"], self.params["b5"])
        self.layers["Relu5"] = Relu()
        self.layers["Dropout1"] = Dropout(dropout_ratio)
        self.layers["Affine2"] = Affine(self.params["W6"], self.params["b6"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        weight_decay = 0
        for idx in range(1, 7):
            W = self.params[f"W{idx}"]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            y = self.predict(x[start:end], train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == t[start:end])

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in range(1, 7):
            grads[f"W{idx}"] = numerical_gradient(loss_w, self.params[f"W{idx}"])
            grads[f"b{idx}"] = numerical_gradient(loss_w, self.params[f"b{idx}"])

        return grads

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = self.last_layer.backward(1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {
            "W1": self.layers["Conv1"].dW,
            "b1": self.layers["Conv1"].db,
            "W2": self.layers["Conv2"].dW,
            "b2": self.layers["Conv2"].db,
            "W3": self.layers["Conv3"].dW,
            "b3": self.layers["Conv3"].db,
            "W4": self.layers["Conv4"].dW,
            "b4": self.layers["Conv4"].db,
            "W5": self.layers["Affine1"].dW,
            "b5": self.layers["Affine1"].db,
            "W6": self.layers["Affine2"].dW,
            "b6": self.layers["Affine2"].db,
        }

        for idx in range(1, 7):
            grads[f"W{idx}"] += self.weight_decay_lambda * self.params[f"W{idx}"]

        return grads

    def save_params(self, file_name="params.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        layer_keys = ["Conv1", "Conv2", "Conv3", "Conv4", "Affine1", "Affine2"]
        for i, key in enumerate(layer_keys, start=1):
            self.layers[key].W = self.params[f"W{i}"]
            self.layers[key].b = self.params[f"b{i}"]
