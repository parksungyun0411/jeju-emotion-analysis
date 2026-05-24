import pandas as pd
import re

# merged_jeju_dedup.xlsx 파일 읽기
df = pd.read_excel("merged_jeju_dedup.xlsx")

print("🔍 한 셀에 여러 문장이 있는 경우 찾기")
print("=" * 50)

# 문장 구분자 패턴 (. ? !)
sentence_pattern = r'[.!?]'

multi_sentence_rows = []

for index, row in df.iterrows():
    text = str(row['표준어 문장'])
    
    # 문장 구분자로 나누기
    sentences = re.split(sentence_pattern, text)
    # 빈 문자열 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 1:
        multi_sentence_rows.append({
            'index': index,
            'original_text': text,
            'sentences': sentences,
            'sentence_count': len(sentences)
        })

print(f"📊 총 {len(df):,}행 중에서 여러 문장이 있는 행: {len(multi_sentence_rows):,}개")

if multi_sentence_rows:
    print("\n🔍 처음 10개 예시:")
    for i, item in enumerate(multi_sentence_rows[:10]):
        print(f"\n행 {item['index']+1}:")
        print(f"원본: {item['original_text']}")
        print(f"문장 수: {item['sentence_count']}개")
        for j, sentence in enumerate(item['sentences']):
            print(f"  {j+1}. {sentence}")
else:
    print("❌ 여러 문장이 있는 행을 찾을 수 없습니다.")
