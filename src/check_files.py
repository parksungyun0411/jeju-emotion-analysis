import pandas as pd
import os

# 두 파일 확인
files = ["최종데이터.xlsx", "최종데이터_균형.xlsx"]

for file in files:
    if os.path.exists(file):
        print(f"\n📊 {file} 파일 구조 확인")
        print("=" * 50)
        
        df = pd.read_excel(file)
        print(f"📏 총 행 수: {len(df):,}개")
        print(f"📋 컬럼 수: {len(df.columns)}개")
        print(f"📝 컬럼명: {list(df.columns)}")
        
        print(f"\n🔍 처음 3행 데이터:")
        print(df.head(3))
        
        # 감정별 분포 확인
        if 'gpt감정' in df.columns:
            emotion_counts = df['gpt감정'].value_counts()
            print(f"\n📈 감정별 분포:")
            for emotion, count in emotion_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {emotion}: {count:,}개 ({percentage:.1f}%)")
        
        if '감정번호' in df.columns:
            emotion_num_counts = df['감정번호'].value_counts().sort_index()
            print(f"\n🔢 감정번호별 분포:")
            for emotion_num, count in emotion_num_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {emotion_num}: {count:,}개 ({percentage:.1f}%)")
    else:
        print(f"❌ {file} 파일을 찾을 수 없습니다.")
