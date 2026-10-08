"""
TROPOMI(S5P) NO2 L2 → 동아시아(20-50N, 100-145E) 슬라이싱 → parquet.

GES DISC 미러(S5P_L2__NO2____HiR v2)를 기존 Earthdata ~/.netrc로 접근 — CDSE 인증 불필요.
granule 하나가 570MB이지만 **전체 다운로드하지 않고** earthaccess + h5py의 HTTP range
부분읽기로 필요한 변수만 뽑는다(동아시아는 전체 4172 스캔라인 중 ~770줄뿐).

보존 항목:
  - time_utc: 스캔라인별 실제 관측시각. NO2는 단명·고변동이라 모든 feature가
    이 순간에 매칭돼야 한다(일평균 금지 원칙).
  - pixel_area_km2: 화소 4코너에서 계산. 스와스 가장자리 화소가 nadir의 ~3배라
    가중 없는 단순합은 화소밀도에 대한 비일치 추정량이 되어 궤도축(105°) 줄무늬를
    만든다. 이후 구적가중에 사용.

주의: h5py 직접 읽기는 CF 스케일을 적용하지 않는다. qa_value는 uint8(0-100) 저장이라
스케일 없이 qa>0.75로 거르면 사실상 전부 통과한다. 아래 _cf_read가 수동 적용한다.

사용:
  python scripts/fetch/slice_tropomi.py 2019-12-29 2020-01-31

주의 (QA P5, 2026-09-24): 이 스크립트는 구 v1 슬라이서(출력 data/raw/tropomi_east_asia, 열 time_utc·latitude·longitude·no2_trop_column·qa_value …).
5년 인덱스(build_index)의 입력은 NAS `_tropomi_ea_v2`(열 lat·lon·qa·obs_time·no2_tvcd, 다른 스크립트 산출)이며 이 출력과 스키마가 다르다 — 이 출력을 build_index 에 넣지 않는다.
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

OUT_DIR = "data/raw/tropomi_east_asia"

LAT_MIN, LAT_MAX = 20.0, 50.0
LON_MIN, LON_MAX = 100.0, 145.0

QA_MIN = 0.75  # TROPOMI 권장 임계(구름·눈얼음·에러 픽셀 제거)
# ↑ 셋은 CLI(--lon-max / --qa-min / --out-dir)로 덮어쓸 수 있다.
#   qa_min=0 으로 받으면 품질 미달 화소가 보존돼 결측 메커니즘 표본이 된다
#   (기본 0.75로 받으면 통과분만 남아 L_masked가 0이 된다).
N_WORKERS = 6  # 네트워크 바운드(CPU 8%) — 동시 granule 수

SHORT_NAME = "S5P_L2__NO2____HiR"
VERSION = "2"

R_EARTH_KM = 6371.0


def _cf_read(dset, sl):
    """h5py 데이터셋을 CF 규약(scale_factor·add_offset·_FillValue)대로 디코딩해 읽는다."""
    raw = dset[sl]
    a = dset.attrs
    out = np.asarray(raw, dtype=np.float64)
    fill = a.get("_FillValue")
    if fill is not None:
        out = np.where(np.asarray(raw) == np.asarray(fill).ravel()[0], np.nan, out)
    scale = a.get("scale_factor")
    offset = a.get("add_offset")
    if scale is not None:
        out = out * np.asarray(scale).ravel()[0]
    if offset is not None:
        out = out + np.asarray(offset).ravel()[0]
    return out


def _pixel_area_km2(lat_b, lon_b):
    """화소 4코너 → 면적[km²]. 국소 등장방형 근사(경도는 cos(lat)로 축소) + 신발끈 공식."""
    lat0 = np.nanmean(lat_b, axis=-1, keepdims=True)
    x = np.deg2rad(lon_b) * R_EARTH_KM * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat_b) * R_EARTH_KM
    xs, ys = np.roll(x, -1, axis=-1), np.roll(y, -1, axis=-1)
    return 0.5 * np.abs(np.sum(x * ys - xs * y, axis=-1))


def slice_one(granule, out_dir=None, qa_min=None, lon_max=None) -> str:
    import earthaccess
    import h5py

    warnings.filterwarnings("ignore")
    out_dir = OUT_DIR if out_dir is None else out_dir
    qa_min = QA_MIN if qa_min is None else qa_min
    lon_max = LON_MAX if lon_max is None else lon_max
    name = granule["meta"]["native-id"].split(":")[-1].replace(".nc", "")
    out_path = os.path.join(out_dir, f"{name}.parquet")
    if os.path.exists(out_path):
        return f"{name}: 이미 존재 (스킵)"

    earthaccess.login(strategy="netrc")
    fh = earthaccess.open([granule])[0]
    with h5py.File(fh, "r") as f:
        p = f["PRODUCT"]
        lat = p["latitude"][0]
        lon = p["longitude"][0]

        in_box = (
            (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= lon_max)
        )
        rows = np.where(in_box.any(axis=1))[0]
        if len(rows) == 0:
            return f"{name}: 도메인 미통과 (스킵)"

        sl = slice(int(rows.min()), int(rows.max()) + 1)
        lat, lon, in_box = lat[sl], lon[sl], in_box[sl]

        no2 = _cf_read(p["nitrogendioxide_tropospheric_column"], (0, sl))
        no2_prec = _cf_read(p["nitrogendioxide_tropospheric_column_precision"], (0, sl))
        qa = _cf_read(p["qa_value"], (0, sl))
        amf = _cf_read(p["air_mass_factor_troposphere"], (0, sl))
        scan_time = p["time_utc"][0, sl]  # 스캔라인별 관측시각

        g = f["PRODUCT/SUPPORT_DATA/GEOLOCATIONS"]
        lat_b = _cf_read(g["latitude_bounds"], (0, sl))
        lon_b = _cf_read(g["longitude_bounds"], (0, sl))
        sza = _cf_read(g["solar_zenith_angle"], (0, sl))

    mask = in_box & (qa >= qa_min) & np.isfinite(no2)
    n = int(mask.sum())
    if n == 0:
        return f"{name}: QA 통과 관측 0건 (스킵)"

    area = _pixel_area_km2(lat_b, lon_b)
    scan_idx = np.broadcast_to(np.arange(lat.shape[0])[:, None], lat.shape)[mask]
    times = pd.to_datetime(
        pd.Series([t.decode() if isinstance(t, bytes) else t for t in scan_time]),
        format="ISO8601",
        utc=True,
    ).to_numpy()

    df = pd.DataFrame(
        {
            "time_utc": times[scan_idx],
            "latitude": lat[mask].astype(np.float32),
            "longitude": lon[mask].astype(np.float32),
            "no2_trop_column": no2[mask].astype(np.float32),  # mol m-2
            "no2_trop_column_precision": no2_prec[mask].astype(np.float32),
            "qa_value": qa[mask].astype(np.float32),
            "amf_troposphere": amf[mask].astype(np.float32),
            "solar_zenith_angle": sza[mask].astype(np.float32),
            "pixel_area_km2": area[mask].astype(np.float32),
        }
    )
    df.to_parquet(out_path + ".tmp", index=False); os.replace(out_path + ".tmp", out_path)  # 원자적: 존재 = 완성본 (건너뛰기 캐시 전제, QA P2)
    return f"{name}: {n}건 (QA통과율 {mask.sum() / in_box.sum():.1%}) → {os.path.basename(out_path)}"


LIMIT = 0


def _check_params(out_dir: str, qa_min: float, lon_max: float) -> None:
    """건너뛰기 캐시는 파일명(granule)만 본다 → 같은 out_dir 을 다른 qa_min·lon_max 로 재사용하면 이전 조건의 파일이 섞인다 (QA D6).
    out_dir/_slice_params.json 에 조건을 기록하고, 기록과 다르면 중단한다. 기록이 없던 기존 디렉토리는 이번 조건을 기록(이전 조건은 검증 불가)."""
    import json
    p = os.path.join(out_dir, "_slice_params.json"); cur = {"qa_min": float(qa_min), "lon_max": float(lon_max)}
    if os.path.exists(p):
        old = json.load(open(p))
        if old != cur:
            raise ValueError(f"{out_dir} 는 {old} 조건으로 만들어짐 ≠ 요청 {cur} — --out-dir 을 바꾸세요")
    else:
        json.dump(cur, open(p, "w"))


def main(start: str, end: str):
    global OUT_DIR, QA_MIN, LON_MAX, N_WORKERS, LIMIT
    import earthaccess

    os.makedirs(OUT_DIR, exist_ok=True); _check_params(OUT_DIR, QA_MIN, LON_MAX)
    earthaccess.login(strategy="netrc")
    granules = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(start, end),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    if LIMIT: granules = granules[:LIMIT]
    print(f"{start} ~ {end}: granule {len(granules)}개, worker {N_WORKERS}")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(slice_one, g, OUT_DIR, QA_MIN, LON_MAX): g for g in granules}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                print(f"  [{i}/{len(granules)}] {fut.result()}", flush=True)
            except Exception as e:  # granule 단위 실패는 건너뛰고 계속
                name = futs[fut]["meta"]["native-id"].split(":")[-1]
                print(f"  [{i}/{len(granules)}] {name}: 실패 — {e}", flush=True)

    print(f"완료 → {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("start"); ap.add_argument("end")
    ap.add_argument("--lon-max", type=float, default=LON_MAX)
    ap.add_argument("--qa-min", type=float, default=QA_MIN)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="granule 수 제한(시간 측정용)")
    a = ap.parse_args()
    LON_MAX = a.lon_max; QA_MIN = a.qa_min; OUT_DIR = a.out_dir; N_WORKERS = a.workers
    LIMIT = a.limit
    print(f"설정: lon≤{LON_MAX} · qa≥{QA_MIN} · out={OUT_DIR} · workers={N_WORKERS}"
          + (f" · limit={LIMIT}" if LIMIT else ""))
    main(a.start, a.end)
