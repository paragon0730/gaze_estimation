"""MPIIGaze 시선 추정 파이프라인. 교재 common/ 모듈(numpy) 위에 만든다.

- data.py     : 라벨/이미지 읽기, 캐시, fold 분할, 표준화
- layers.py   : 회귀용으로 새로 추가한 계층 (Concat, MeanSquaredLoss)
- model.py    : GazeLeNet (교재 SimpleConvNet 확장, Zhang et al. 2015 구조)
- trainer.py  : 미니배치 학습 루프, 검증 오차 기록
- metrics.py  : 각도 오차
"""
