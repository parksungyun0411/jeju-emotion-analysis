import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings('ignore')

def analyze_emotion_performance(df, text_column, dataset_name, text_type):
    """
    감정별 상세 성능 분석 함수
    """
    print(f"\n📊 {dataset_name} ({text_type}) 감정별 상세 성능 분석")
    print("=" * 80)
    
    # 텍스트 전처리
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        import re
        text = str(text).strip()
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    df_processed = df.copy()
    df_processed['processed_text'] = df_processed[text_column].apply(preprocess_text)
    df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),
        stop_words=None,
        lowercase=False,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True
    )
    
    X = vectorizer.fit_transform(df_processed['processed_text'])
    y = df_processed['감정번호'].values
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 감정 라벨
    emotion_labels = {
        0: "중립",
        1: "슬픔", 
        2: "행복",
        3: "분노",
        4: "놀람",
        5: "공포",
        6: "혐오"
    }
    
    # 모델들
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'SVM': SVC(random_state=42, kernel='linear', probability=True),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Naive Bayes': MultinomialNB(alpha=1.0)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🤖 {model_name} 분석 중...")
        
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 감정별 상세 성능 지표
        report = classification_report(
            y_test, y_pred, 
            target_names=list(emotion_labels.values()),
            output_dict=True,
            zero_division=0
        )
        
        # 결과 저장
        results[model_name] = {
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # 상세 결과 출력
        print(f"\n📋 {model_name} 감정별 상세 성능:")
        print("-" * 70)
        print(f"{'감정':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        for emotion_num, emotion_name in emotion_labels.items():
            if emotion_name in report:
                precision = report[emotion_name]['precision']
                recall = report[emotion_name]['recall']
                f1 = report[emotion_name]['f1-score']
                support = report[emotion_name]['support']
                
                print(f"{emotion_name:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}")
        
        # 전체 성능
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        
        print("-" * 70)
        print(f"{'Accuracy':<8} {accuracy:<10.4f} {'':<10} {'':<10} {len(y_test):<10.0f}")
        print(f"{'Macro Avg':<8} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {macro_avg['support']:<10.0f}")
        print(f"{'Weighted Avg':<8} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {weighted_avg['support']:<10.0f}")
    
    return results

def save_detailed_results(jeju_results, standard_results):
    """
    상세 결과를 텍스트 파일로 저장
    """
    with open('감정별_상세성능_분석결과.txt', 'w', encoding='utf-8') as f:
        f.write("🎯 감정별 상세 성능 분석 결과 (Precision, Recall, F1-Score, Support)\n")
        f.write("=" * 100 + "\n\n")
        
        # 제주어 결과
        f.write("📊 제주어 학습 결과\n")
        f.write("=" * 50 + "\n\n")
        
        for dataset_name, results in jeju_results.items():
            f.write(f"🔸 {dataset_name}.xlsx (제주어 데이터)\n")
            f.write("-" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"🤖 {model_name}\n")
                f.write("-" * 30 + "\n")
                
                report = result['classification_report']
                emotion_labels = {
                    0: "중립", 1: "슬픔", 2: "행복", 3: "분노", 
                    4: "놀람", 5: "공포", 6: "혐오"
                }
                
                f.write(f"{'감정':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
                f.write("-" * 50 + "\n")
                
                for emotion_num, emotion_name in emotion_labels.items():
                    if emotion_name in report:
                        precision = report[emotion_name]['precision']
                        recall = report[emotion_name]['recall']
                        f1 = report[emotion_name]['f1-score']
                        support = report[emotion_name]['support']
                        
                        f.write(f"{emotion_name:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}\n")
                
                # 전체 성능
                accuracy = report['accuracy']
                macro_avg = report['macro avg']
                weighted_avg = report['weighted avg']
                
                f.write("-" * 50 + "\n")
                f.write(f"{'Accuracy':<8} {accuracy:<10.4f} {'':<10} {'':<10} {len(result['y_test']):<10.0f}\n")
                f.write(f"{'Macro Avg':<8} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {macro_avg['support']:<10.0f}\n")
                f.write(f"{'Weighted Avg':<8} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {weighted_avg['support']:<10.0f}\n\n")
        
        # 표준어 결과
        f.write("\n📊 표준어 학습 결과\n")
        f.write("=" * 50 + "\n\n")
        
        for dataset_name, results in standard_results.items():
            f.write(f"🔸 {dataset_name}.xlsx (표준어 데이터)\n")
            f.write("-" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"🤖 {model_name}\n")
                f.write("-" * 30 + "\n")
                
                report = result['classification_report']
                emotion_labels = {
                    0: "중립", 1: "슬픔", 2: "행복", 3: "분노", 
                    4: "놀람", 5: "공포", 6: "혐오"
                }
                
                f.write(f"{'감정':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
                f.write("-" * 50 + "\n")
                
                for emotion_num, emotion_name in emotion_labels.items():
                    if emotion_name in report:
                        precision = report[emotion_name]['precision']
                        recall = report[emotion_name]['recall']
                        f1 = report[emotion_name]['f1-score']
                        support = report[emotion_name]['support']
                        
                        f.write(f"{emotion_name:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}\n")
                
                # 전체 성능
                accuracy = report['accuracy']
                macro_avg = report['macro avg']
                weighted_avg = report['weighted avg']
                
                f.write("-" * 50 + "\n")
                f.write(f"{'Accuracy':<8} {accuracy:<10.4f} {'':<10} {'':<10} {len(result['y_test']):<10.0f}\n")
                f.write(f"{'Macro Avg':<8} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {macro_avg['support']:<10.0f}\n")
                f.write(f"{'Weighted Avg':<8} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {weighted_avg['support']:<10.0f}\n\n")
        
        # 비교 분석
        f.write("\n📊 제주어 vs 표준어 성능 비교 분석\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("🔍 주요 발견사항:\n\n")
        f.write("1. 감정별 성능 차이 분석\n")
        f.write("2. 모델별 성능 특성 비교\n")
        f.write("3. 언어별 분류 난이도 차이\n")
        f.write("4. 데이터 불균형의 영향\n\n")
        
        f.write("📈 결론:\n")
        f.write("- 제주어와 표준어의 감정 분류 성능을 상세히 비교 분석했습니다.\n")
        f.write("- 각 감정별로 precision, recall, f1-score, support를 확인할 수 있습니다.\n")
        f.write("- 모델별, 데이터셋별 성능 차이를 명확히 파악할 수 있습니다.\n")

def main():
    """
    메인 실행 함수
    """
    print("🎯 감정별 상세 성능 분석 (Precision, Recall, F1-Score, Support)")
    print("=" * 80)
    
    # 데이터셋 목록
    datasets = {
        '최종데이터': '최종데이터.xlsx',
        '최종데이터_균형': '최종데이터_균형.xlsx'
    }
    
    jeju_results = {}
    standard_results = {}
    
    # 제주어 데이터 분석
    print("\n📊 제주어 데이터 분석 시작...")
    for dataset_name, filename in datasets.items():
        try:
            df = pd.read_excel(filename)
            print(f"✅ {filename} 로드 완료: {len(df):,}개 행")
            
            results = analyze_emotion_performance(
                df, '제주어 문장', dataset_name, '제주어'
            )
            jeju_results[dataset_name] = results
            
        except Exception as e:
            print(f"❌ {filename} 제주어 분석 중 오류: {e}")
    
    # 표준어 데이터 분석
    print("\n📊 표준어 데이터 분석 시작...")
    for dataset_name, filename in datasets.items():
        try:
            df = pd.read_excel(filename)
            print(f"✅ {filename} 로드 완료: {len(df):,}개 행")
            
            results = analyze_emotion_performance(
                df, '표준어 문장', dataset_name, '표준어'
            )
            standard_results[dataset_name] = results
            
        except Exception as e:
            print(f"❌ {filename} 표준어 분석 중 오류: {e}")
    
    # 결과 저장
    save_detailed_results(jeju_results, standard_results)
    
    print(f"\n🏁 감정별 상세 성능 분석 완료!")
    print("📂 생성된 파일:")
    print("   - 감정별_상세성능_분석결과.txt : 상세 성능 지표 결과")

if __name__ == "__main__":
    main()
