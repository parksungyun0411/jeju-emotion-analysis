import os
import pandas as pd
import re
import requests
import json
import time
from typing import List, Tuple, Dict
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSentenceProcessor:
    def __init__(self, api_key: str = None):
        """
        여러 문장이 있는 셀에서 가장 감정이 강한 문장만 남기는 클래스
        """
        # API 키 설정
        if api_key:
            self.api_key = api_key
        else:
            try:
                from config import OPENAI_API_KEY
                self.api_key = OPENAI_API_KEY
            except ImportError:
                self.api_key = None
            
            if not self.api_key:
                raise ValueError("OpenAI API 키가 필요합니다. config.py 파일에 OPENAI_API_KEY를 설정해주세요.")
        
        # 감정 분류 카테고리
        self.emotions = {
            0: "중립",
            1: "슬픔", 
            2: "행복",
            3: "분노",
            4: "놀람",
            5: "공포",
            6: "혐오"
        }
        
        # 감정 강도 점수 (중립이 아닌 감정일수록 높은 점수)
        self.emotion_scores = {
            0: 0,   # 중립
            1: 3,   # 슬픔
            2: 4,   # 행복
            3: 5,   # 분노
            4: 2,   # 놀람
            5: 4,   # 공포
            6: 3    # 혐오
        }
        
        self.model = "gpt-4o"
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def classify_emotion(self, text: str) -> Tuple[int, str]:
        """
        OpenAI API를 사용하여 텍스트의 감정을 분류하는 함수
        """
        try:
            prompt = f"""
다음 한국어 텍스트의 감정을 매우 정확하게 분석하여 다음 7가지 감정 중 하나로 분류해주세요.

감정 분류 기준:
0: 중립 - 완전히 감정이 없는 객관적 사실 서술, 단순 정보 전달만
1: 슬픔 - 우울, 아쉬움, 실망, 좌절, 그리움, 외로움, 비애
2: 행복 - 즐거움, 기쁨, 만족, 흥미, 희망, 긍정적 기대, 만족감
3: 분노 - 화남, 짜증, 불만, 적대감, 공격적 의도, 격분
4: 놀람 - 깜짝, 당황, 신기함, 예상 밖의 상황, 충격, 경이로움
5: 공포 - 무서움, 두려움, 불안, 걱정, 위협감, 불안정감
6: 혐오 - 싫음, 역겨움, 거부감, 불쾌함, 혐오감, 기피

텍스트: "{text}"

중요한 분류 원칙:
- 중립(0)은 정말 감정이 전혀 없는 경우에만 사용하세요
- 미묘한 감정이라도 있으면 해당 감정으로 분류하세요
- 긍정적인 내용은 행복(2)으로 분류하세요
- 부정적인 내용은 슬픔(1)으로 분류하세요
- 텍스트의 톤, 어조, 단어 선택을 세심하게 분석하세요
- 한국어의 뉘앙스와 문화적 맥락을 고려하세요

응답 형식: 숫자만 답변해주세요 (예: 2)
"""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "당신은 한국어 텍스트의 감정을 매우 정확하게 분류하는 전문가입니다. 한국어의 뉘앙스와 문화적 맥락을 깊이 이해하고 있습니다."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 5,
                "temperature": 0.0,
                "top_p": 1.0
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 숫자만 추출
                emotion_num = int(re.findall(r'\d+', content)[0])
                
                # 유효한 범위인지 확인
                if emotion_num not in self.emotions:
                    logger.warning(f"예상치 못한 감정 번호: {emotion_num}, 중립으로 설정")
                    emotion_num = 0
                
                emotion_name = self.emotions[emotion_num]
                
                return emotion_num, emotion_name
            else:
                logger.error(f"API 호출 실패: {response.status_code} - {response.text}")
                return 0, "중립"
            
        except Exception as e:
            logger.error(f"감정 분류 중 오류 발생: {e}")
            return 0, "중립"
    
    def split_sentence_pairs(self, jeju_text: str, standard_text: str) -> List[Tuple[str, str]]:
        """
        제주어와 표준어 문장을 함께 . ? ! 기준으로 나누어 쌍으로 만드는 함수
        """
        # 문장 구분자 패턴
        sentence_pattern = r'[.!?]'
        
        # 제주어와 표준어를 각각 나누기
        jeju_sentences = re.split(sentence_pattern, jeju_text)
        standard_sentences = re.split(sentence_pattern, standard_text)
        
        # 빈 문자열 제거하고 앞뒤 공백 제거
        jeju_sentences = [s.strip() for s in jeju_sentences if s.strip()]
        standard_sentences = [s.strip() for s in standard_sentences if s.strip()]
        
        # 문장 수가 다르면 더 긴 쪽에 맞춰서 처리
        max_len = max(len(jeju_sentences), len(standard_sentences))
        
        sentence_pairs = []
        for i in range(max_len):
            jeju_sentence = jeju_sentences[i] if i < len(jeju_sentences) else ""
            standard_sentence = standard_sentences[i] if i < len(standard_sentences) else ""
            
            if jeju_sentence and standard_sentence:
                sentence_pairs.append((jeju_sentence, standard_sentence))
        
        return sentence_pairs
    
    def find_best_sentence_pair(self, sentence_pairs: List[Tuple[str, str]]) -> Tuple[str, str, int, str]:
        """
        여러 문장 쌍 중에서 가장 감정이 강한 문장 쌍을 찾는 함수
        """
        if not sentence_pairs:
            return "", "", 0, "중립"
            
        best_jeju_sentence = sentence_pairs[0][0]  # 기본값
        best_standard_sentence = sentence_pairs[0][1]  # 기본값
        best_emotion_num = 0
        best_emotion_name = "중립"
        best_score = 0
        
        for jeju_sentence, standard_sentence in sentence_pairs:
            if len(standard_sentence.strip()) < 2:  # 너무 짧은 문장은 스킵
                continue
                
            # 표준어 문장으로 감정 분석 (더 정확한 분석을 위해)
            emotion_num, emotion_name = self.classify_emotion(standard_sentence)
            score = self.emotion_scores[emotion_num]
            
            # 감정 강도가 더 높은 문장 쌍을 선택
            if score > best_score:
                best_jeju_sentence = jeju_sentence
                best_standard_sentence = standard_sentence
                best_emotion_num = emotion_num
                best_emotion_name = emotion_name
                best_score = score
            
            # API 호출 제한을 위한 대기
            time.sleep(0.1)
        
        return best_jeju_sentence, best_standard_sentence, best_emotion_num, best_emotion_name
    
    def process_excel_file(self, input_file: str, output_file: str = None):
        """
        Excel 파일을 읽어서 여러 문장이 있는 셀에서 가장 감정이 강한 문장만 남기는 함수
        """
        logger.info(f"Excel 파일 처리 시작: {input_file}")
        
        # Excel 파일 읽기
        try:
            df = pd.read_excel(input_file)
            logger.info(f"Excel 파일 읽기 완료: {len(df)}행")
        except Exception as e:
            logger.error(f"Excel 파일 읽기 실패: {e}")
            return
        
        # 출력 파일명 설정
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_단일문장처리완료.xlsx"
        
        # 여러 문장이 있는 행들 찾기
        multi_sentence_rows = []
        
        for index, row in df.iterrows():
            jeju_text = str(row['제주어 문장'])
            standard_text = str(row['표준어 문장'])
            
            # NaN 값 처리
            if jeju_text == 'nan' or standard_text == 'nan':
                continue
                
            sentence_pairs = self.split_sentence_pairs(jeju_text, standard_text)
            
            if len(sentence_pairs) > 1:
                multi_sentence_rows.append({
                    'index': index,
                    'original_jeju_text': jeju_text,
                    'original_standard_text': standard_text,
                    'sentence_pairs': sentence_pairs
                })
        
        logger.info(f"여러 문장이 있는 행: {len(multi_sentence_rows)}개")
        
        # 처리 통계
        processed_count = 0
        total_rows = len(multi_sentence_rows)
        start_time = datetime.now()
        
        print(f"\n🚀 다중 문장 처리 시작!")
        print(f"📊 처리할 총 행 수: {total_rows:,}개")
        print(f"⏰ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 사용 모델: {self.model}")
        print("=" * 50)
        
        # 각 행에 대해 처리
        for i, item in enumerate(multi_sentence_rows):
            try:
                index = item['index']
                sentence_pairs = item['sentence_pairs']
                
                # 가장 감정이 강한 문장 쌍 찾기
                best_jeju_sentence, best_standard_sentence, emotion_num, emotion_name = self.find_best_sentence_pair(sentence_pairs)
                
                # 원본 데이터 업데이트
                df.at[index, '제주어 문장'] = best_jeju_sentence
                df.at[index, '표준어 문장'] = best_standard_sentence
                df.at[index, '감정번호'] = emotion_num
                df.at[index, 'gpt감정'] = emotion_name
                
                processed_count += 1
                
                # 진행률 계산
                progress = (processed_count / total_rows) * 100
                elapsed_time = datetime.now() - start_time
                
                # 진행 상황 로깅 (매 50개마다)
                if processed_count % 50 == 0:
                    estimated_total_time = elapsed_time * (total_rows / processed_count)
                    remaining_time = estimated_total_time - elapsed_time
                    
                    print(f"📈 진행률: {progress:.1f}% ({processed_count:,}/{total_rows:,})")
                    print(f"⏱️ 경과 시간: {elapsed_time}")
                    print(f"🕐 예상 남은 시간: {remaining_time}")
                    print(f"📝 현재 처리 중: {best_standard_sentence[:50]}...")
                    print(f"😊 분류 결과: {emotion_name}")
                    print("-" * 30)
                
            except Exception as e:
                logger.error(f"행 {item['index']} 처리 중 오류: {e}")
                print(f"❌ 오류 발생: 행 {item['index']} - {e}")
                continue
        
        # 완료 시간 계산
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # Excel 파일로 저장
        print(f"\n💾 Excel 파일로 저장 중: {output_file}")
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"\n✅ 처리 완료!")
        print(f"📁 저장된 파일: {output_file}")
        print(f"📊 처리된 문장 수: {processed_count:,}개")
        print(f"⏰ 총 소요 시간: {total_time}")
        print(f"⚡ 평균 처리 속도: {processed_count/total_time.total_seconds():.2f} 문장/초")
        
        # 감정별 분포 출력
        emotion_counts = df['gpt감정'].value_counts()
        print(f"\n📈 감정별 분포:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {emotion}: {count:,}개 ({percentage:.1f}%)")
        
        print("=" * 50)
        
        return df

def main():
    """
    메인 실행 함수
    """
    try:
        # API 키 확인
        try:
            from config import OPENAI_API_KEY
            if not OPENAI_API_KEY:
                print("config.py 파일에 OPENAI_API_KEY를 설정해주세요.")
                return
        except ImportError:
            print("config.py 파일을 찾을 수 없습니다. config.py 파일을 생성하고 OPENAI_API_KEY를 설정해주세요.")
            return
        
        # 입력 파일 경로
        input_file = "merged_jeju_dedup.xlsx"
        
        if not os.path.exists(input_file):
            print(f"입력 파일을 찾을 수 없습니다: {input_file}")
            return
        
        # 다중 문장 처리기 초기화
        processor = MultiSentenceProcessor()
        
        print("=" * 60)
        print("🎯 다중 문장 처리 프로그램")
        print("=" * 60)
        print("📋 처리 내용:")
        print("   - 한 셀에 여러 문장이 있는 경우 감정 분석")
        print("   - 가장 감정이 강한 문장만 남기고 나머지 삭제")
        print("   - 문장 구분: . ? ! 기준")
        print("   - 제주어와 표준어 쌍을 함께 처리")
        print("=" * 60)
        
        # Excel 파일 처리
        result_df = processor.process_excel_file(input_file)
        
        print(f"\n✅ 처리 완료!")
        print(f"📁 생성된 파일: merged_jeju_dedup_단일문장처리완료.xlsx")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
