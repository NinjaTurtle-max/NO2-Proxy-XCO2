import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 환경 설정 (이 부분만 수정하시면 됩니다)
# ==========================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

BASE_DIR = str(config.DOWNLOADS_DIR)  # 다운로드 메인 경로
YEARS = [2020, 2022, 2023, 2024]        # 분석 대상 연도
SECTORS = ["POWER_INDUSTRY", "IND_COMBUSTION", "TRANSPORT"] # 분석 대상 섹터

# 동아시아 영역 설정
LAT_RANGE = slice(20, 50)
LON_RANGE = slice(100, 150)

# 결과 저장용 폴더
OUTPUT_DIR = str(config.EDGAR_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 핵심 처리 함수
# ==========================================
def process_edgar_pipeline():
    all_summary = []

    for sector in SECTORS:
        print(f"\n🚀 섹터 처리 시작: {sector}")
        for year in YEARS:
            # EDGAR 폴더 및 파일명 규칙에 따른 경로 생성
            folder_path = os.path.join(BASE_DIR, f"bkl_{sector}_flx_nc")
            file_name = f"EDGAR_2025_GHG_CO2_{year}_bkl_{sector}_flx.nc"
            full_path = os.path.join(folder_path, file_name)

            if not os.path.exists(full_path):
                print(f"   ⚠️ 파일을 찾을 수 없음: {file_name}")
                continue

            # 데이터 로드 및 슬라이싱
            with xr.open_dataset(full_path, chunks={'time': 1}) as ds:
                ds_ea = ds.sel(lat=LAT_RANGE, lon=LON_RANGE)
                df = ds_ea.to_dataframe().reset_index()
                
                # 유효 데이터 필터링 (0 이상)
                flux_col = [col for col in df.columns if 'flux' in col.lower()][0]
                df_valid = df[df[flux_col] > 0].dropna()
                
                # 결과 저장
                output_name = f"EA_{year}_{sector}.csv"
                df_valid.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
                
                # 통계 요약 기록
                total_flux = df_valid[flux_col].sum()
                all_summary.append({'Year': year, 'Sector': sector, 'Total_Flux': total_flux})
                print(f"   ✅ {year}년 완료: {len(df_valid)} 행 추출")

    # 요약 리포트 저장
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Processing_Summary.csv"), index=False)
    return summary_df

# ==========================================
# 3. 데이터 정합성 진단 (2020 vs 2024 산점도)
# ==========================================
def plot_integrity_check(summary_df):
    for sector in SECTORS:
        path_20 = os.path.join(OUTPUT_DIR, f"EA_2020_{sector}.csv")
        path_24 = os.path.join(OUTPUT_DIR, f"EA_2024_{sector}.csv")
        
        if os.path.exists(path_20) and os.path.exists(path_24):
            df_20 = pd.read_csv(path_20)
            df_24 = pd.read_csv(path_24)
            
            # 특정 월(예: 1월) 데이터만 추출해서 비교
            m20 = df_20[df_20['time'].str.contains('-01-')]
            m24 = df_24[df_24['time'].str.contains('-01-')]
            
            # 좌표 기준 병합
            merged = pd.merge(m20, m24, on=['lat', 'lon'], suffixes=('_20', '_24'))
            
            plt.figure(figsize=(8, 8))
            plt.scatter(merged['fluxes_24'], merged['fluxes_20'], alpha=0.3, s=10)
            plt.plot([merged['fluxes_24'].min(), merged['fluxes_24'].max()],
                     [merged['fluxes_24'].min(), merged['fluxes_24'].max()], 'r--', label='y=x (No Change)')
            
            plt.xscale('log'); plt.yscale('log')
            plt.title(f"Integrity Check: {sector} (Jan 2020 vs 2024)")
            plt.xlabel("2024 Flux (Normal)")
            plt.ylabel("2020 Flux (COVID-19)")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.savefig(os.path.join(OUTPUT_DIR, f"Integrity_Check_{sector}.png"))
            plt.close()

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    summary = process_edgar_pipeline()
    plot_integrity_check(summary)
    print("\n✨ 모든 작업이 완료되었습니다. 'EDGAR_Processed_Results' 폴더를 확인하세요.")
