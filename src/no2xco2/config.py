"""경로 설정 — 하드코딩 금지 원칙. 환경변수로 덮어쓸 수 있다.

NAS_ROOT : NAS 프로젝트 루트. 결정 순서 (사용자 결정 ①, 2026-10-08 — 공개 저장소에 내부 주소를 남기지 않음):
  ① 환경변수 NO2_NAS_ROOT ② 로컬 파일 configs/nas_local.txt 첫 줄 (git 제외, 견본 configs/nas_local.example.txt)
  ③ 자리표시 /Volumes/NAS/dataset/NO2-Proxy-XCO2
NO2_NAS_MOUNT : NAS 공유 루트 = "dataset" 의 상위 (기본: NAS_ROOT 에서 "/dataset/" 앞부분 — /Volumes/<호스트> 또는 PC 의 E:, 없으면 NAS_ROOT)
  — 마운트 대기(wait_nas)·로컬 판정(_LOCAL_NAS)·SMB 가드(era5.open_wind)·NAS_SRC_NC·smb URL 이 모두 이 값에서 나온다 (QA A3 2026-09-24).
"""
import os

_NAS_LOCAL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "nas_local.txt")
_NAS_PLACEHOLDER = "/Volumes/NAS/dataset/NO2-Proxy-XCO2"


def _nas_root() -> str:
    """NAS_ROOT 결정: 환경변수 NO2_NAS_ROOT → configs/nas_local.txt 첫 줄 → 자리표시 (빈 값은 다음 단계로)."""
    env = os.environ.get("NO2_NAS_ROOT", "").strip()
    if env:
        return env
    if os.path.isfile(_NAS_LOCAL_FILE):
        with open(_NAS_LOCAL_FILE, encoding="utf-8") as f:
            line = f.readline().strip()
        if line:
            return line
    return _NAS_PLACEHOLDER


NAS_ROOT = _nas_root()
NAS_MOUNT = os.environ.get("NO2_NAS_MOUNT", NAS_ROOT.split("/dataset/")[0] if "/dataset/" in NAS_ROOT else NAS_ROOT)  # nas_xco2_dir·NAS_SRC_NC 가 <NAS_MOUNT>/dataset/… 를 찾으므로 "dataset" 상위 (QA K2)
NAS_ERA5_RAW = os.path.join(NAS_ROOT, "era5_pl_raw")       # 기압면·단일면 원본 (보존)
NAS_ERA5_WIND = os.path.join(NAS_ROOT, "era5_wind_z100")    # 파생 PBL풍 (z_min=100m)
NAS_TROPOMI_PQ = os.path.join(NAS_ROOT, "tropomi_parquet")  # NAS CSV → 날짜별 parquet
LOCAL_STAGE_RAW = "data/raw/era5_pl_tmp"                    # 로컬 스테이징 (월 1건 ~1.3GB)
LOCAL_STAGE_OUT = "data/raw/era5_wind"
NAS_OCO_NODES = os.path.join(NAS_ROOT, "oco_nodes")          # OCO 사운딩 노드 테이블 (year_month 파티션)
NAS_SRC_NC = os.path.join(NAS_MOUNT, "dataset", "XCO2연구 데이터", "integrated_dataset.nc")
LOCAL_NC = "data/raw/integrated_dataset.nc"                   # 처리 중 로컬 사본 (SMB 랜덤 I/O 회피)
GRID_LAT0, GRID_LON0, GRID_D = 20.0, 100.0, 0.25              # 잠재 격자 = ERA5 격자 (121×201)
INDEX_DIR = "data/processed/index"                             # build_index 산출 (5년 인덱스)
ERA5_SYN_DIR = "data/raw/era5_syn"                             # ERA5 기압면 파생 z850·thk (era5_syn.py)

# 3-3 소규모 조달 (rung 0 대조·TCCON 외부검증·ODIAC 보충)
NAS_TCCON = os.path.join(NAS_ROOT, "tccon_ggg2020")
NAS_CT = os.path.join(NAS_ROOT, "carbontracker_ea")
NAS_CAMS = os.path.join(NAS_ROOT, "cams_ea")
NAS_ODIAC2025 = os.path.join(NAS_ROOT, "odiac2025")
LOCAL_STAGE_DL = "data/raw/dl_tmp"

# NAS PC(Windows)에서 직접 실행하면 NAS 루트가 로컬 드라이브 → 마운트 대기 불필요. NO2_NAS_MOUNT 재지정도 반영 (QA K2)
_LOCAL_NAS = os.name == "nt" or not NAS_MOUNT.startswith("/Volumes/")


def wait_nas(max_wait_s: int = 6 * 3600, poll_s: int = 60) -> None:
    """SMB 마운트가 끊긴 동안 대기. 마운트 루트가 없으면 poll_s 간격으로 재확인. 로컬 드라이브면 존재만 확인."""
    import time
    if _LOCAL_NAS:
        if not os.path.isdir(os.path.dirname(NAS_ROOT)):
            raise OSError(f"NAS 루트 상위 디렉토리 없음: {NAS_ROOT} (NO2_NAS_ROOT 확인)")
        return
    t0 = time.time()
    while not os.path.isdir(os.path.join(NAS_MOUNT, "dataset")):
        if time.time() - t0 > max_wait_s:
            raise OSError(f"NAS 마운트 없음 {max_wait_s}s 초과: {NAS_MOUNT}")
        print(f"    NAS 마운트 없음 → 재접속 시도 후 {poll_s}s 대기", flush=True)
        import subprocess
        if sys_platform() == "darwin":
            subprocess.run(["open", f"smb://{os.path.basename(NAS_MOUNT.rstrip('/'))}/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(poll_s)


def sys_platform() -> str:
    import sys; return sys.platform


def _same_ends(a: str, b: str, n: int = 4 << 20) -> bool:
    """두 파일의 앞·뒤 n 바이트가 같은가 (크기 대조만으로 내용을 보증하지 않던 문제, QA K4). 크기가 다르면 False."""
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    size = os.path.getsize(a)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        if fa.read(n) != fb.read(n):
            return False
        fa.seek(max(size - n, 0)); fb.seek(max(size - n, 0))
        return fa.read(n) == fb.read(n)


def _part_prefix_ok(src: str, part: str, n: int = 4 << 20) -> bool:
    """이어쓰기 전 .part 가 src 의 앞부분인지 확인 — .part 의 마지막 n 바이트를 src 같은 위치와 대조 (QA K4)."""
    done = os.path.getsize(part)
    if done > os.path.getsize(src):
        return False
    off = max(done - n, 0)
    with open(src, "rb") as fs, open(part, "rb") as fp:
        fs.seek(off); fp.seek(off)
        return fs.read(done - off) == fp.read(done - off)


def move_to_nas(src: str, dst: str, tries: int = 20) -> None:
    """로컬 → NAS 이동: 마운트 대기 → .part 복사 → 크기 대조 → rename → 로컬 삭제. EIO 시 재시도.
    shutil.move/_fcopyfile은 SMB 끊김·과부하에서 EIO로 죽고 조각 파일을 남긴다."""
    import time
    part = dst + ".part"
    for k in range(1, tries + 1):
        try:
            wait_nas()
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and _same_ends(src, dst):  # 이전 시도에서 rename까지 끝난 경우 (크기 + 앞·뒤 4 MB 대조, QA K4)
                os.remove(src); return
            if os.path.exists(part) and not _part_prefix_ok(src, part):  # 다른 파일의 잔여 .part 에 이어쓰지 않는다 (QA K4)
                os.remove(part)
            done = os.path.getsize(part) if os.path.exists(part) else 0  # 끊긴 자리부터 이어쓰기 (1 GB 파일이 SMB 순단마다 처음부터 가지 않도록)
            with open(src, "rb") as fi, open(part, "ab") as fo:
                fi.seek(done)
                while True:
                    b = fi.read(4 << 20)
                    if not b: break
                    fo.write(b)
            if os.path.getsize(part) != os.path.getsize(src):
                raise OSError(f"size mismatch {os.path.getsize(part)} != {os.path.getsize(src)}")
            os.replace(part, dst)
            os.remove(src)
            return
        except OSError as e:
            print(f"    NAS 이동 실패({k}/{tries}) {os.path.basename(dst)}: {e} — {min(60*k,300)}s 후 재시도(이어쓰기)", flush=True)
            if "size mismatch" in str(e):
                try: os.remove(part)
                except OSError: pass
            time.sleep(min(60 * k, 300))
    raise OSError(f"NAS 이동 {tries}회 실패: {dst}")


def nas_xco2_dir() -> str:
    """NAS의 'XCO2연구 데이터' 디렉토리. SMB가 한글 이름을 NFD로 돌려주므로 정규화 비교로 찾는다."""
    import unicodedata
    parent = os.path.join(NAS_MOUNT, "dataset")
    for e in os.scandir(parent):
        if unicodedata.normalize("NFC", e.name) == "XCO2연구 데이터":
            return e.path
    raise FileNotFoundError(f"'XCO2연구 데이터' 없음: {parent}")


# TROPOMI 원천 (2026-09-08 전환): granule별 parquet 8,011개, 2020-01-01~2024-12-31, qa≥0.66 화소, 화소별 obs_time, 40칼럼.
# CSV(qa≥0.75, 8칼럼, granule 시각)→parquet 변환(06)은 중단. 접근은 nas_tropomi_ea_v2() 로.
def nas_tropomi_ea_v2() -> str:
    return os.path.join(nas_xco2_dir(), "_tropomi_ea_v2")


def nas_makedirs(path: str, tries: int = 6) -> None:
    """NAS 경로 makedirs: 마운트 대기 후 시도. 마운트가 빠진 순간 /Volumes/<ip> 자체를 만들려다 PermissionError 가 나는 것을 막는다."""
    import time
    for k in range(1, tries + 1):
        try:
            wait_nas(); os.makedirs(path, exist_ok=True); return
        except OSError as e:
            print(f"    NAS makedirs 실패({k}/{tries}) {path}: {e} — {60*k}s 후 재시도", flush=True); time.sleep(60 * k)
    raise OSError(f"NAS makedirs {tries}회 실패: {path}")
