import requests
import json
import pandas as pd
import os
from PIL import Image

# 단계 1: 데이터 수집 및 저장
def load_data_from_api():
    url = "https://ll.thespacedevs.com/2.0.0/launch/upcoming"
    save_folder = "data_storage"
    file_path = f"{save_folder}/launches.json"
    
    os.makedirs(save_folder, exist_ok=True)
    
    print("🛰️ 데이터를 가져오는 중...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return None
    
    # JSON 파일 저장
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    # 데이터 구조 안전하게 읽기 (에러 방지용)
    results = data.get('results', [])
    if not results:
        print("⚠️ 데이터가 비어있습니다.")
        return data

    df = pd.DataFrame(results)
    print("--- 데이터 구조 확인 ---")
    print(df.head())
    
    return data

# 단계 2: 이미지 다운로드 및 팝업
def get_pictures(data):
    if not data: return
    
    img_folder = "downloaded_images"
    os.makedirs(img_folder, exist_ok=True)
    
    launches = data.get('results', [])
    
    for i, launch in enumerate(launches):
        img_url = launch.get('image')
        
        if not img_url:
            continue
            
        try:
            # 이미지 다운로드
            img_data = requests.get(img_url, timeout=10).content
            file_name = f"{img_folder}/launch_{i}.jpg"
            
            with open(file_name, "wb") as f:
                f.write(img_data)
            
            print(f"✅ [{i}] 저장 및 확인 중: {file_name}")
            
            # 윈도우 기본 사진 앱으로 띄우기
            img = Image.open(file_name)
            img.show()
            
            # 3개만 보고 멈춤
            if i >= 2:
                print("\n🚀 처음 3개 이미지를 확인했습니다. 다운로드를 종료합니다.")
                break
                
        except Exception as e:
            print(f"❌ [{i}] 이미지 처리 중 에러: {e}")

# 실행부
if __name__ == "__main__":
    launch_data = load_data_from_api()
    if launch_data:
        get_pictures(launch_data)
    print("\n모든 작업이 완료되었습니다!")
