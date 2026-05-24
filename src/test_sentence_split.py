import pandas as pd
import re

# 테스트용 작은 샘플 데이터 생성
test_data = {
    '제주어 문장': [
        '아 진짜? 그거 왜 언니 안보내줬어?',
        '그거 이게 그거 뭐야? 뭐에 좋은거야?',
        '와이프는 있나? 아들은 하나 있대'
    ],
    '표준어 문장': [
        '아 진짜? 그거 왜 언니 안보내줬어?',
        '그거 이게 그거 뭐야? 뭐에 좋은거야?',
        '와이프는 있나? 아들은 하나 있대'
    ],
    '감정번호': [0, 0, 0],
    'gpt감정': ['중립', '중립', '중립']
}

df = pd.DataFrame(test_data)

# 문장 구분자 패턴
sentence_pattern = r'[.!?]'

print("🔍 테스트 데이터 확인")
print("=" * 50)

for index, row in df.iterrows():
    jeju_text = str(row['제주어 문장'])
    standard_text = str(row['표준어 문장'])
    
    # 제주어와 표준어를 각각 나누기
    jeju_sentences = re.split(sentence_pattern, jeju_text)
    standard_sentences = re.split(sentence_pattern, standard_text)
    
    # 빈 문자열 제거하고 앞뒤 공백 제거
    jeju_sentences = [s.strip() for s in jeju_sentences if s.strip()]
    standard_sentences = [s.strip() for s in standard_sentences if s.strip()]
    
    print(f"\n행 {index+1}:")
    print(f"제주어 원본: {jeju_text}")
    print(f"표준어 원본: {standard_text}")
    print(f"제주어 문장 수: {len(jeju_sentences)}개")
    print(f"표준어 문장 수: {len(standard_sentences)}개")
    
    for i, (jeju_sent, standard_sent) in enumerate(zip(jeju_sentences, standard_sentences)):
        print(f"  {i+1}. 제주어: {jeju_sent}")
        print(f"     표준어: {standard_sent}")
