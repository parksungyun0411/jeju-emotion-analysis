import pandas as pd
import os

# merged_jeju_dedup.xlsx 파일 읽기
try:
    df = pd.read_excel("merged_jeju_dedup.xlsx")
    print("📊 파일 구조 확인")
    print("=" * 50)
    print(f"📁 파일명: merged_jeju_dedup.xlsx")
    print(f"📏 총 행 수: {len(df):,}개")
    print(f"📋 컬럼 수: {len(df.columns)}개")
    print(f"📝 컬럼명: {list(df.columns)}")
    print("\n🔍 처음 5행 데이터:")
    print(df.head())
    
    print("\n🔍 샘플 데이터 (10행):")
    for i in range(min(10, len(df))):
        print(f"행 {i+1}: {df.iloc[i].to_dict()}")
        
except Exception as e:
    print(f"❌ 파일 읽기 실패: {e}")
