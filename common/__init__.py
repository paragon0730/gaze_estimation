# coding: utf-8
"""교재 『밑바닥부터 시작하는 딥러닝』 공통 모듈 (https://github.com/ssuai/deep_learning_from_scratch/tree/master/common).

원본을 그대로 복사했고, 아래 세 군데만 고쳤다 (`[수정]` 주석으로 표시).

- util.py `im2col`, `col2im`: 중간 배열을 입력과 같은 dtype으로 만든다.
  원본은 항상 float64를 만들어 float32로 학습해도 연산이 float64로 바뀐다. 결과 값은 같다.
- layers.py `Pooling.backward`: 같은 이유로 dtype을 맞춘다.
- functions.py `step_function`: numpy 2.x에서 없어진 `np.int` 대신 `int`.

원본 라이선스는 LICENSE.md (MIT, Koki Saitoh).
"""
