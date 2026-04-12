import os
import gzip
import shutil
import tarfile
from pathlib import Path

# ==============================================================================
# ODIAC 데이터 압축 해제 스크립트 (WSL 환경 E드라이브 대응)
# ==============================================================================

# 윈도우의 E 드라이브에 저장하시려면 WSL 경로인 /mnt/e/ 를 사용하셔야 합니다.
# (E:／... 같이 입력하면 리눅스 시스템 내부에 가상 폴더가 생겨 디스크 용량이 꽉 찹니다)
OUTPUT_DIR = Path(r"/mnt/e/extracted_odiac_2020_2023")

def extract_data(start_year=2020, end_year=2023):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"압축 해제된 파일은 다음 폴더에 저장됩니다:\n{OUTPUT_DIR}\n")
    
    # 윈도우 E드라이브는 WSL에서 보통 /mnt/e 로 매핑됩니다.
    base_wsl_path = Path("/mnt/e")
    
    for year in range(start_year, end_year + 1):
        file_name = f"odiac2024_1km_excl_intl_{year}_allmonths.tif.gz"
        target_path = base_wsl_path / file_name
        
        if not target_path.exists():
            print(f"[{year}년] 파일을 찾을 수 없습니다: {target_path}")
            continue
            
        print(f"[{year}년] 압축 해제 시도: {target_path.name}")
        
        # 1. 경로 자체가 폴더인 경우 (예: 압축해제 프로그램이 폴더명으로 풀어놓은 경우)
        if target_path.is_dir():
            print(f"  - 해당 경로가 폴더입니다. 내부의 '{year}' 폴더를 탐색합니다.")
            
            subfolder = target_path / str(year)
            if not subfolder.exists() or not subfolder.is_dir():
                print(f"    - '{year}' 하위 폴더를 찾을 수 없거나 폴더가 아닙니다: {subfolder}")
                continue
                
            yy = str(year)[-2:] # 2020 -> 20, 2021 -> 21
            for month in range(1, 13):
                month_str = f"{month:02d}"
                expected_prefix = f"odiac2024_1km_excl_intl_{yy}{month_str}"
                
                # 파일 찾기 (확장자가 .tif 이지만 실제로는 zip일 수 있음)
                match_files = list(subfolder.glob(f"*{expected_prefix}*"))
                
                if not match_files:
                    print(f"    - 데이터를 찾을 수 없습니다: {expected_prefix}")
                    continue
                    
                for file_path in match_files:
                    # 저장할 파일명 설정 (.zip이나 .gz가 있다면 제거)
                    if str(file_path.name).endswith('.zip'):
                        output_filename = str(file_path.name)[:-4]
                    elif str(file_path.name).endswith('.gz'):
                        output_filename = str(file_path.name)[:-3]
                    else:
                        output_filename = str(file_path.name)
                        
                    output_path = OUTPUT_DIR / output_filename
                    print(f"    - 처리 중: {file_path.name}")
                    
                    try:
                        import zipfile
                        # 윈도우에서 '압축된 보관 폴더'로 나오는데 확장자가 .tif라면 내부적으로 zip 파일일 가능성이 높음
                        if zipfile.is_zipfile(file_path):
                            with zipfile.ZipFile(file_path, "r") as zip_ref:
                                zip_ref.extractall(OUTPUT_DIR)
                            print(f"      -> zip 압축 해제 완료")
                        else:
                            # gzip 시도
                            try:
                                with gzip.open(file_path, 'rb') as f_in:
                                    with open(output_path, 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                                print(f"      -> gzip 압축 해제 완료: {output_filename}")
                            except gzip.BadGzipFile:
                                # 이미 압축이 풀린 일반 .tif 파일일 경우 단순 복사
                                shutil.copy2(file_path, output_path)
                                print(f"      -> 압축되지 않은 일반 파일로 확인되어 복사 완료: {output_filename}")
                    except Exception as e:
                        print(f"    - 오류 발생 ({file_path.name}): {e}")
            continue

        # 2. 만약 실제로는 tar 아카이브(여러 달의 파일이 묶여있음)인 경우
        is_tar_file = False
        try:
            with tarfile.open(target_path, "r:gz") as tar:
                print(f"  - 여러 파일이 포함된 tar.gz 형식으로 확인됨. 압축 해제 중...")
                tar.extractall(path=OUTPUT_DIR)
                is_tar_file = True
        except tarfile.ReadError:
            is_tar_file = False
            
        # 3. 단일 gzip 파일(.tif.gz)인 경우
        if not is_tar_file:
            print(f"  - 단일 파일 형식(.tif.gz)으로 확인됨. 압축 해제 중...")
            output_filename = str(target_path.name)[:-3] # .gz 제거
            output_path = OUTPUT_DIR / output_filename
            
            try:
                with gzip.open(target_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  - 완료: {output_filename}")
            except Exception as e:
                print(f"  - 오류 발생: {e}")

if __name__ == "__main__":
    extract_data(2020, 2023)
    print("\n모든 압축 해제 작업이 완료되었습니다.")
