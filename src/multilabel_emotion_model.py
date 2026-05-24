import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class EmotionClassifier:
    def __init__(self, max_features=50000, ngram_range=(1, 3)):
        """
        N-gram 기반 감정 분류 모델 클래스
        
        Args:
            max_features: TF-IDF 벡터화 시 최대 특성 수
            ngram_range: N-gram 범위 (1,3) = 단어, 2-gram, 3-gram
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # 한국어는 불용어 처리를 별도로 해야함
            lowercase=False,  # 한국어는 대소문자 구분 안함
            sublinear_tf=True,  # sublinear TF 스케일링 적용
            use_idf=True,
            smooth_idf=True
        )
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'SVM': SVC(
                random_state=42,
                kernel='linear',
                probability=True
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'Naive Bayes': MultinomialNB(
                alpha=1.0
            )
        }
        
        self.trained_models = {}
        self.emotion_labels = {
            0: "중립",
            1: "슬픔", 
            2: "행복",
            3: "분노",
            4: "놀람",
            5: "공포",
            6: "혐오"
        }
        
    def preprocess_text(self, text):
        """
        텍스트 전처리 함수
        """
        if pd.isna(text):
            return ""
        
        # 기본 정리
        text = str(text).strip()
        
        # 특수문자 일부 제거 (문장 부호는 유지)
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def prepare_data(self, df, text_column='표준어 문장', emotion_column='감정번호'):
        """
        데이터 준비 함수
        """
        print(f"📊 데이터 준비 중...")
        print(f"   - 원본 데이터: {len(df):,}개")
        
        # 텍스트 전처리
        df_processed = df.copy()
        df_processed['processed_text'] = df_processed[text_column].apply(self.preprocess_text)
        
        # 빈 텍스트 제거
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        
        print(f"   - 전처리 후 데이터: {len(df_processed):,}개")
        
        # 감정별 분포 확인
        emotion_dist = df_processed[emotion_column].value_counts().sort_index()
        print(f"\n📈 감정별 분포:")
        for emotion_num, count in emotion_dist.items():
            emotion_name = self.emotion_labels[emotion_num]
            percentage = (count / len(df_processed)) * 100
            print(f"   {emotion_num}: {emotion_name} - {count:,}개 ({percentage:.1f}%)")
        
        return df_processed
    
    def train_models(self, df_processed, text_column='processed_text', emotion_column='감정번호'):
        """
        여러 모델을 학습시키는 함수
        """
        print(f"\n🚀 모델 학습 시작!")
        print("=" * 60)
        
        # 텍스트 벡터화
        print("📝 TF-IDF 벡터화 중...")
        X = self.vectorizer.fit_transform(df_processed[text_column])
        y = df_processed[emotion_column].values
        
        print(f"   - 벡터 차원: {X.shape}")
        
        # 훈련/테스트 분할 (80:20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   - 훈련 데이터: {X_train.shape[0]:,}개")
        print(f"   - 테스트 데이터: {X_test.shape[0]:,}개")
        
        # 각 모델 학습 및 평가
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🤖 {model_name} 학습 중...")
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 성능 평가
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   ✅ F1-macro: {f1_macro:.4f}")
            print(f"   ✅ F1-weighted: {f1_weighted:.4f}")
            
            # 학습된 모델 저장
            self.trained_models[model_name] = model
        
        return results, X_test, y_test, X_train, X_test
    
    def generate_report(self, results, X_test, y_test, dataset_name):
        """
        상세한 성능 평가 보고서 생성
        """
        print(f"\n📊 {dataset_name} 성능 보고서")
        print("=" * 80)
        
        # 전체 성능 요약
        print("🏆 모델별 전체 성능")
        print("-" * 60)
        performance_summary = []
        
        for model_name, result in results.items():
            performance_summary.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'F1-Macro': f"{result['f1_macro']:.4f}",
                'F1-Weighted': f"{result['f1_weighted']:.4f}"
            })
        
        summary_df = pd.DataFrame(performance_summary)
        print(summary_df.to_string(index=False))
        
        # 최고 성능 모델 찾기
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🥇 최고 성능 모델: {best_model_name}")
        print(f"   - Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"   - F1-macro: {results[best_model_name]['f1_macro']:.4f}")
        
        # 상세 분류 보고서
        print(f"\n📋 {best_model_name} 상세 분류 보고서")
        print("-" * 60)
        
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred']
        
        # 문자열 라벨로 변환
        y_test_labels = [self.emotion_labels[i] for i in y_test]
        y_pred_labels = [self.emotion_labels[i] for i in y_pred]
        
        # 분류 보고서 생성
        report = classification_report(
            y_test_labels, y_pred_labels, 
            target_names=list(self.emotion_labels.values()),
            output_dict=True
        )
        
        # 보고서를 DataFrame으로 변환
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(4)
        
        print(report_df.to_string())
        
        # 혼동 행렬 생성
        self.plot_confusion_matrix(y_test, y_pred, best_model_name, dataset_name)
        
        # 모델 저장
        self.save_model(best_model, best_model_name, dataset_name)
        
        # 결과 저장
        self.save_results(results, dataset_name)
        
        return best_model_name, results[best_model_name]
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, dataset_name):
        """
        혼동 행렬 시각화
        """
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # 라벨 이름 배열 생성
        labels = [self.emotion_labels[i] for i in range(7)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        
        plt.title(f'{dataset_name} - {model_name} 혼동행렬')
        plt.xlabel('예측 라벨')
        plt.ylabel('실제 라벨')
        
        # 그래프 저장
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{dataset_name}_{model_name}_confusion_matrix.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 혼동 행렬 저장: results/{dataset_name}_{model_name}_confusion_matrix.png")
    
    def save_model(self, model, model_name, dataset_name):
        """
        모델 저장
        """
        os.makedirs('models', exist_ok=True)
        
        # 모델과 벡터라이저 함께 저장
        model_data = {
            'model': model,
            'vectorizer': self.vectorizer,
            'emotion_labels': self.emotion_labels,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f'models/{dataset_name}_{model_name}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 모델 저장됨: {filename}")
    
    def save_results(self, results, dataset_name):
        """
        성능 결과 저장
        """
        os.makedirs('results', exist_ok=True)
        
        # 성능 요약 저장
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'F1_Macro': result['f1_macro'],
                'F1_Weighted': result['f1_weighted'],
                'Test_Size': len(result['y_test'])
            })
        
        performance_df = pd.DataFrame(performance_data)
        filename = f'results/{dataset_name}_performance_summary.csv'
        performance_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"📊 모델 간 기존 오류 내용 제거 후 성능 비교 저장된 파일: {filename}")

def main():
    """
    메인 실행 함수
    """
    print("🎯 N-gram 기반 감정 분류 모델 학습 및 평가")
    print("=" * 80)
    
    # 데이터셋 목록
    datasets = {
        '최종데이터': '최종데이터.xlsx',
        '최종데이터_균형': '최종데이터_균형.xlsx'
    }
    
    for dataset_name, filename in datasets.items():
        print(f"\n📁 {dataset_name} 처리 중...")
        print("=" * 60)
        
        try:
            # 데이터 로드
            df = pd.read_excel(filename)
            print(f"✅ 데이터 로드 완료: {len(df):,}개 행")
            
            # 감정 분류기 초기화
            classifier = EmotionClassifier(
                max_features=30000,  # 메모리 고려하여 조정
                ngram_range=(1, 3)   # 1-gram부터 3-gram까지
            )
            
            # 데이터 준비
            df_processed = classifier.prepare_data(df)
            
            # 모델 학습
            results, X_test, y_test, X_train, X_test = classifier.train_models(df_processed)
            
            # 성능 보고서 생성
            best_model_name, best_result = classifier.generate_report(results, X_test, y_test, dataset_name)
            
            print(f"\n✅ {dataset_name} 처리 완료!")
            print(f"   최고 성능 모델: {best_model_name}")
            print(f"   Accuracy: {best_result['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ {dataset_name} 처리 중 오류 발생: {e}")
            continue
    
    print(f"\n🏁 모든 데이터셋 처리 완료!")
    print("📂 생성된 파일들:")
    print("   - models/ : 학습된 모델들")
    print("   - results/ : 학습된 모델들과 성능 비교 저장된 CSV 그리고 혼동행렬 이미지 이미지의 파일들")

if __name__ == "__main__":
    main()
