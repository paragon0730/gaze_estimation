"""회귀 문제를 위해 교재 계층(common/layers.py)에 더한 계층. 교재와 같은 forward/backward 형식."""
import numpy as np


class Concat:
    """완전연결 특징 x (N, D)와 head pose h (N, 2)를 이어 붙인다.

    backward는 덧셈 노드처럼 기울기를 그대로 나눠 준다 (5장 계산 그래프).
    """

    def __init__(self):
        self.d = None

    def forward(self, x, h):
        self.d = x.shape[1]
        return np.concatenate([x, h], axis=1)

    def backward(self, dout):
        return dout[:, :self.d], dout[:, self.d:]


class MeanSquaredLoss:
    """항등 함수 출력 + 평균 제곱 오차. SoftmaxWithLoss의 회귀 버전.

    L = 1/N · Σ_n ½‖y_n − t_n‖²,  ∂L/∂y = (y − t) / N
    """

    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, y, t):
        self.y, self.t = y, t
        self.loss = 0.5 * np.sum((y - t) ** 2) / y.shape[0]
        return self.loss

    def backward(self, dout=1):
        return dout * (self.y - self.t) / self.y.shape[0]
