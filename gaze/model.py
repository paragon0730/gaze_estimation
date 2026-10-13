"""GazeLeNet: 교재 7장 SimpleConvNet을 시선 회귀용으로 확장한 CNN.

구조는 MPIIGaze 논문(Zhang et al., CVPR 2015)의 GazeNet(LeNet 기반)을 따른다.

    eye (1,36,60) ─ conv(20,5×5) ─ relu ─ pool(2) ─ conv(50,5×5) ─ relu ─ pool(2)
                  ─ affine(500) ─ relu ─ [dropout] ─┐
    head (2) ───────────────────────────────────────┴─ concat ─ affine(2) ─ MSE

SimpleConvNet과 다른 점
- 합성곱 계층 2개 (5장·7장의 계층을 그대로 씀)
- 마지막 완전연결 계층 앞에서 head pose를 이어 붙임 (gaze.layers.Concat)
- 분류(SoftmaxWithLoss) 대신 회귀(gaze.layers.MeanSquaredLoss)
- He 초깃값 (6장), float32
"""
import pickle
from collections import OrderedDict

import numpy as np

from common.layers import Affine, Convolution, Dropout, Pooling, Relu
from gaze.layers import Concat, MeanSquaredLoss


class GazeLeNet:
    def __init__(self, input_dim=(1, 36, 60), conv_params=((20, 5), (50, 5)), hidden_size=500,
                 head_size=2, output_size=2, dropout_ratio=0.0, seed=None, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.dropout_ratio = dropout_ratio
        self.params = {}
        self.conv_names = []

        # 합성곱 블록: conv - relu - pool(2×2)
        C, H, W = input_dim
        for i, (fn, fs) in enumerate(conv_params, start=1):
            self.params[f"W{i}"] = rng.standard_normal((fn, C, fs, fs)) * np.sqrt(2.0 / (C * fs * fs))
            self.params[f"b{i}"] = np.zeros(fn)
            C, H, W = fn, (H - fs + 1) // 2, (W - fs + 1) // 2
            self.conv_names.append(i)
        self.feature_shape = (C, H, W)
        n_conv = len(conv_params)

        # 완전연결: flatten → hidden,  [hidden + head] → output
        flat = C * H * W
        self.params[f"W{n_conv + 1}"] = rng.standard_normal((flat, hidden_size)) * np.sqrt(2.0 / flat)
        self.params[f"b{n_conv + 1}"] = np.zeros(hidden_size)
        # 출력층은 교재 기본값처럼 작게(0.01) 시작해 초기 예측이 평균 gaze 근처에 오게 한다
        self.params[f"W{n_conv + 2}"] = rng.standard_normal((hidden_size + head_size, output_size)) * 0.01
        self.params[f"b{n_conv + 2}"] = np.zeros(output_size)
        for k in self.params:
            self.params[k] = self.params[k].astype(dtype)

        # 계층 생성 (교재와 같은 OrderedDict 방식)
        self.layers = OrderedDict()
        for i in self.conv_names:
            self.layers[f"Conv{i}"] = Convolution(self.params[f"W{i}"], self.params[f"b{i}"], stride=1, pad=0)
            self.layers[f"Relu{i}"] = Relu()
            self.layers[f"Pool{i}"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params[f"W{n_conv + 1}"], self.params[f"b{n_conv + 1}"])
        self.layers["ReluFC"] = Relu()
        if dropout_ratio > 0:
            self.layers["Dropout"] = Dropout(dropout_ratio)
        self.concat = Concat()
        self.out_layer = Affine(self.params[f"W{n_conv + 2}"], self.params[f"b{n_conv + 2}"])
        self.last_layer = MeanSquaredLoss()
        self._param_layers = [(f"Conv{i}", i) for i in self.conv_names] + \
                             [("Affine1", n_conv + 1)]

    # ---- forward ----
    def features(self, x, train_flg=False):
        """concat 직전까지의 특징 (N, hidden)."""
        for layer in self.layers.values():
            x = layer.forward(x, train_flg) if isinstance(layer, Dropout) else layer.forward(x)
        return x

    def predict(self, x, head, train_flg=False):
        h = self.features(x, train_flg)
        return self.out_layer.forward(self.concat.forward(h, head))

    def loss(self, x, head, t, train_flg=False):
        return self.last_layer.forward(self.predict(x, head, train_flg), t)

    def predict_batches(self, x, head, batch_size=500):
        return np.concatenate([self.predict(x[i:i + batch_size], head[i:i + batch_size])
                               for i in range(0, len(x), batch_size)])

    # ---- backward (오차역전파) ----
    def gradient(self, x, head, t):
        loss = self.loss(x, head, t, train_flg=True)

        dout = self.last_layer.backward(1)
        dout = self.out_layer.backward(dout)
        dout, _ = self.concat.backward(dout)  # head pose는 학습 대상이 아니므로 기울기를 버린다
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        for name, i in self._param_layers:
            grads[f"W{i}"], grads[f"b{i}"] = self.layers[name].dW, self.layers[name].db
        n = len(self._param_layers) + 1
        grads[f"W{n}"], grads[f"b{n}"] = self.out_layer.dW, self.out_layer.db
        return loss, grads

    def numerical_gradient(self, x, head, t):
        """수치 미분 (4장). 작은 망에서 gradient()를 검증할 때만 쓴다."""
        from common.gradient import numerical_gradient
        f = lambda _: self.loss(x, head, t)
        return {k: numerical_gradient(f, v) for k, v in self.params.items()}

    # ---- 저장/불러오기 (교재와 같은 pickle 방식) ----
    def save_params(self, file_name="params.pkl", extra=None):
        with open(file_name, "wb") as f:
            pickle.dump({"params": self.params, "extra": extra or {}}, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            obj = pickle.load(f)
        self.set_params(obj["params"])
        return obj.get("extra", {})

    def set_params(self, params):
        """같은 배열 객체를 계층과 공유하므로 제자리(in-place)로 값을 바꾼다."""
        for k, v in params.items():
            self.params[k][...] = v

    def copy_params(self):
        return {k: v.copy() for k, v in self.params.items()}
