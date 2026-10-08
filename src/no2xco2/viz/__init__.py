"""결과 시각화 공통 (GIS/Visualization 담당, 2026-09-22).

- 팔레트: dataviz 기준 범주형 고정 순서 (slot 1 파랑 = 우리 물리, 2 주황 = CT, 3 청록 = 절제). 시리즈에 색을 순환 배정하지 않는다.
- 발산형(지도의 ± 값)은 범주색을 쓰지 않는다 (QA B3): 보라(−) – 중립 회색 – 적갈(+). 범주 slot 1·2 와 겹치지 않게 별도 토큰.
- 마크: 선 2 px · 마커 ≥ 8 px(흰 테두리 2 px) · 막대 ≤ 24 px · 격자선 1 px 회색.
- 출력: docs/results/<이름>_<날짜>.png (300 dpi) + 같은 이름 .csv (그림에 쓴 수치).
- 백엔드·전역 rcParams 는 여기서 바꾸지 않는다 (노트북 inline 유지). 스타일은 styled 데코레이터가 rc_context 로 그림 함수 안에서만 적용 (QA W8).
- 저장은 기존 파일을 덮어쓰지 않는다 (N-2 생성일 규약, QA W1). 덮어쓰기·접미사는 `with save_options(overwrite=..., suffix=...)` 안에서만 (끝나면 원복, QA Y6).
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.environ.get("NO2_RESULTS_DIR", "docs/results")
_OPT = {"overwrite": False, "suffix": ""}  # save_options 로만 바꾼다


class save_options:
    """문맥 안에서만 덮어쓰기 허용·파일명 접미사(예: "_r2" → name_2026-09-24_r2.png) 적용. 나가면 이전 값으로 원복 (QA Y6)."""
    def __init__(self, overwrite: bool = False, suffix: str = ""):
        self.new = {"overwrite": overwrite, "suffix": suffix}

    def __enter__(self):
        self.old = dict(_OPT); _OPT.update(self.new); return self

    def __exit__(self, *exc):
        _OPT.clear(); _OPT.update(self.old)


def out_paths(name: str, date: str, out_dir: str | None = None) -> tuple[str, str]:
    """save() 가 쓸 png·csv 경로 (현재 save_options 반영). 진입점의 사전 충돌 검사용 (QA Y3)."""
    out_dir = out_dir or RESULTS_DIR; sfx = _OPT["suffix"]
    return os.path.join(out_dir, f"{name}_{date}{sfx}.png"), os.path.join(out_dir, f"{name}_{date}{sfx}.csv")
C_OURS, C_CT, C_NOPHYS = "#2a78d6", "#eb6834", "#1baf7a"   # 범주형 slot 1·2·3 (light)
C_GRID, C_TEXT2, C_MUTED = "#d9d8d3", "#52514e", "#a09f98"
C_DIV_NEG, C_DIV_MID, C_DIV_POS = "#5e3c99", "#f4f4f2", "#b2182b"  # 발산형 양 극 + 중립 (범주색과 분리)
LABEL = {"ours_phys": "우리(물리)", "ours_nophys": "절제(물리 제거)", "CT_debiased(train)": "CT(편향 제거)", "CT_raw": "CT(원값)"}
COLOR = {"ours_phys": C_OURS, "ours_nophys": C_NOPHYS, "CT_debiased(train)": C_CT, "CT_raw": C_CT}


RC = {
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_GRID, "axes.linewidth": 1,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": C_GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "xtick.color": C_TEXT2, "ytick.color": C_TEXT2, "axes.labelcolor": C_TEXT2, "text.color": "#0b0b0b",
    "lines.linewidth": 2, "lines.markersize": 8, "lines.markeredgewidth": 2, "lines.markeredgecolor": "white",
    "legend.frameon": False, "legend.handlelength": 3.2, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
}


def styled(fn):
    """그림 함수를 rc_context(RC) 안에서 실행 — 전역 rcParams 를 바꾸지 않는다 (QA W8)."""
    import functools

    @functools.wraps(fn)
    def wrap(*a, **k):
        with plt.rc_context(RC):
            return fn(*a, **k)
    return wrap


def save(fig, table: pd.DataFrame, name: str, date: str, out_dir: str | None = None) -> tuple[str, str]:
    """그림 PNG + 그림에 쓴 수치 CSV 를 같은 이름으로 저장. out_dir None → RESULTS_DIR."""
    out_dir = out_dir or RESULTS_DIR; os.makedirs(out_dir, exist_ok=True)
    png, csv = out_paths(name, date, out_dir)
    if not _OPT["overwrite"] and (os.path.exists(png) or os.path.exists(csv)):
        plt.close(fig)
        raise FileExistsError(f"{png} 이미 있음 — N-2 생성일 규약상 덮어쓰지 않는다. 같은 날 재생성이면 --suffix _r2 등, 의도한 덮어쓰기면 --overwrite")
    fig.savefig(png); plt.close(fig); table.to_csv(csv, index=False)
    return png, csv
