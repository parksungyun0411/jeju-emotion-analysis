import os
import re
import pandas as pd
import requests
import json
import time
import random
from typing import List, Tuple, Dict
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JejuEmotionClassifier:
    def __init__(self, api_key: str = None):
        """
        제주도 감정 분류 자동화 클래스
        
        Args:
            api_key: OpenAI API 키 (config.py에서 자동으로 가져옴)
        """
        # API 키 설정
        if api_key:
            self.api_key = api_key
        else:
            # config.py에서 API 키 가져오기
            try:
                from config import OPENAI_API_KEY
                self.api_key = OPENAI_API_KEY
            except ImportError:
                self.api_key = None
            
            if not self.api_key:
                raise ValueError("OpenAI API 키가 필요합니다. config.py 파일에 OPENAI_API_KEY를 설정해주세요.")
        
        # 감정 분류 카테고리 (7가지)
        self.emotions = {
            0: "중립",
            1: "슬픔", 
            2: "행복",
            3: "분노",
            4: "놀람",
            5: "공포",
            6: "혐오"
        }
        
        # GPT 모델 설정 (가장 좋은 모델 사용)
        self.model = "gpt-4o"  # 최신 GPT-4o 모델 사용 (가장 정확함)
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def classify_emotion(self, text: str) -> Tuple[int, str]:
        """
        OpenAI API를 사용하여 텍스트의 감정을 분류하는 함수
        
        Args:
            text: 감정을 분류할 텍스트
            
        Returns:
            (감정번호, 감정명) 튜플
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
                "temperature": 0.0,  # 가장 일관된 결과를 위해 0으로 설정
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
    
    def process_excel_file(self, input_file: str, output_file: str = None, start_row: int = 0, end_row: int = None):
        """
        Excel 파일을 읽어서 감정 분류를 추가하는 함수
        
        Args:
            input_file: 입력 Excel 파일 경로
            output_file: 출력 Excel 파일 경로 (None이면 자동 생성)
            start_row: 시작 행 번호 (0부터 시작)
            end_row: 끝 행 번호 (None이면 파일 끝까지)
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
            output_file = f"{base_name}_감정분류완료_{start_row}to{end_row}.xlsx"
        
        # 특정 행 범위 선택
        if end_row is None:
            end_row = len(df)
        
        df = df.iloc[start_row:end_row].copy()
        logger.info(f"처리할 행 범위: {start_row}번째부터 {end_row-1}번째까지 ({len(df)}행)")
        
        # 감정 분류 컬럼 초기화
        df['감정번호'] = 0
        df['gpt감정'] = '중립'
        
        # 각 행에 대해 감정 분류 수행
        processed_count = 0
        total_rows = len(df)
        start_time = datetime.now()
        
        print(f"\n🚀 감정 분류 시작!")
        print(f"📊 처리할 총 행 수: {total_rows:,}개")
        print(f"⏰ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 사용 모델: {self.model}")
        print("=" * 50)
        
        for index, row in df.iterrows():
            try:
                # 표준어 문장이 있는 컬럼 찾기 (보통 B열 또는 '표준어' 컬럼)
                text_to_classify = None
                
                # 가능한 컬럼명들 확인
                possible_columns = ['표준어', '표준어 문장', 'B', '문장', 'text']
                for col in possible_columns:
                    if col in df.columns:
                        text_to_classify = str(row[col])
                        break
                
                # 컬럼명을 찾지 못한 경우 두 번째 컬럼 사용
                if text_to_classify is None and len(df.columns) >= 2:
                    text_to_classify = str(row.iloc[1])  # 두 번째 컬럼 (B열)
                
                if text_to_classify and text_to_classify != 'nan':
                    # 감정 분류
                    emotion_num, emotion_name = self.classify_emotion(text_to_classify)
                    
                    # 결과 저장
                    df.at[index, '감정번호'] = emotion_num
                    df.at[index, 'gpt감정'] = emotion_name
                    
                    processed_count += 1
                    
                    # 진행률 계산
                    progress = (processed_count / total_rows) * 100
                    elapsed_time = datetime.now() - start_time
                    
                    # 진행 상황 로깅 (매 10개마다)
                    if processed_count % 10 == 0:
                        estimated_total_time = elapsed_time * (total_rows / processed_count)
                        remaining_time = estimated_total_time - elapsed_time
                        
                        print(f"📈 진행률: {progress:.1f}% ({processed_count:,}/{total_rows:,})")
                        print(f"⏱️ 경과 시간: {elapsed_time}")
                        print(f"🕐 예상 남은 시간: {remaining_time}")
                        print(f"📝 현재 처리 중: {text_to_classify[:50]}...")
                        print(f"😊 분류 결과: {emotion_name}")
                        print("-" * 30)
                    
                    # API 호출 제한을 위한 대기
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"행 {index} 처리 중 오류: {e}")
                print(f"❌ 오류 발생: 행 {index} - {e}")
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
            percentage = (count / processed_count) * 100
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
        input_file = "제주어_표준어_전처리완성1.xlsx"
        
        if not os.path.exists(input_file):
            print(f"입력 파일을 찾을 수 없습니다: {input_file}")
            return
        
        # 감정 분류기 초기화
        classifier = JejuEmotionClassifier()
        
        print("=" * 60)
        print("🎯 제주어 감정분류 자동화 프로그램")
        print("=" * 60)
        print("📋 처리할 행 범위:")
        print("   - 2행부터 9999행까지 (9,998개 문장)")
        print("   - 60000행부터 69999행까지 (10,000개 문장)")
        print("📊 총 처리할 문장 수: 19,998개")
        print("=" * 60)
        
        # 첫 번째 범위: 2행부터 9999행까지 (인덱스 1부터 9998까지)
        print("\n🚀 첫 번째 범위 처리 시작: 2행부터 9999행까지")
        output_file1 = "제주어_표준어_전처리완성1_감정분류_2to9999.xlsx"
        result_df1 = classifier.process_excel_file(input_file, output_file1, start_row=1, end_row=9999)
        
        # 두 번째 범위: 60000행부터 69999행까지 (인덱스 59999부터 69998까지)
        print("\n🚀 두 번째 범위 처리 시작: 60000행부터 69999행까지")
        output_file2 = "제주어_표준어_전처리완성1_감정분류_60000to69999.xlsx"
        result_df2 = classifier.process_excel_file(input_file, output_file2, start_row=59999, end_row=69999)
        
        print(f"\n✅ 모든 범위 처리 완료!")
        print(f"📁 생성된 파일:")
        print(f"   - {output_file1} (2행~9999행)")
        print(f"   - {output_file2} (60000행~69999행)")
        print(f"📊 총 처리된 문장 수: 19,998개")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()