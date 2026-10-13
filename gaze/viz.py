"""노트북 공용 그림 설정과 gaze 화살표 그리기."""
import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

KOREAN_FONTS = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"]

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98"


def setup():
    """그림 스타일. 한글 글꼴이 있으면 쓴다 (Ubuntu/Colab: apt install fonts-nanum)."""
    for path in glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf"):
        font_manager.fontManager.addfont(path)  # 방금 설치한 글꼴은 캐시에 없으므로 직접 등록
    names = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in KOREAN_FONTS if f in names]
    if korean:
        plt.rcParams["font.family"] = korean[0]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]),
        "font.size": 9, "axes.titlesize": 10,
    })


def draw_gaze(ax, img, yaw_pitch, color, label=None, length=None):
    """read_sample.ipynb와 같은 방식: 3D gaze 벡터의 (x, y) 성분 방향으로 화살표."""
    h, w = img.shape
    length = length or w / 2
    dx = -length * np.cos(yaw_pitch[1]) * np.sin(yaw_pitch[0])
    dy = -length * np.sin(yaw_pitch[1])
    ax.arrow(w / 2, h / 2, dx, dy, color=color, width=0.4, head_width=2,
             length_includes_head=True, label=label)
