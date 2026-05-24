import pandas as pd
from datetime import datetime

# 두 데이터셋의 성능 결과를 읽어오기
df_imbalanced = pd.read_csv('results/최종데이터_performance_summary.csv')
df_balanced = pd.read_csv('results/최종데이터_균형_performance_summary.csv')

# 종합 보고서 생성
report = f"""
🎯 N-gram 기반 감정 분류 베이스라인 성능 평가 종합 결과
============================================================

📊 실험 정보:
  - 실험 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  - 벡터화: TF-IDF (1-3 gram, max_features=30,000)
  - 분할: 80% 학습, 20% 테스트
  - 평가 지표: Accuracy, F1-Macro, F1-Weighted

📈 데이터셋별 성능 비교
============================================================

🔸 최종데이터.xlsx (불균형 데이터 - 72,879개)
------------------------------------------------------------
모델                   정확도        F1-Macro      F1-Weighted   
------------------------------------------------------------"""

for _, row in df_imbalanced.iterrows():
    report += f"""
{row['Model']:<20} {row['Accuracy']:.4f}      {row['F1_Macro']:.4f}      {row['F1_Weighted']:.4f}"""

# 최고 성능 모델 찾기
best_imbalanced = df_imbalanced.loc[df_imbalanced['Accuracy'].idxmax()]
report += f"""

🏆 최고 성능 모델: {best_imbalanced['Model']}
🥇 베이스라인 성능: {best_imbalanced['Accuracy']:.4f} (Accuracy), {best_imbalanced['F1_Macro']:.4f} (F1-Macro)

🔸 최종데이터_균형.xlsx (균형 데이터 - 12,950개)
------------------------------------------------------------
모델                   정확도        F1-Macro      F1-Weighted   
------------------------------------------------------------"""

for _, row in df_balanced.iterrows():
    report += f"""
{row['Model']:<20} {row['Accuracy']:.4f}      {row['F1_Macro']:.4f}      {row['F1_Weighted']:.4f}"""

# 최고 성능 모델 찾기
best_balanced = df_balanced.loc[df_balanced['Accuracy'].idxmax()]
report += f"""

🏆 최고 성능 모델: {best_balanced['Model']}
🥇 베이스라인 성능: {best_balanced['Accuracy']:.4f} (Accuracy), {best_balanced['F1_Macro']:.4f} (F1-Macro)

📊 종합 분석
============================================================

🔍 주요 발견사항:
1. 불균형 데이터에서 더 높은 정확도 달성
   - 최종데이터.xlsx: {best_imbalanced['Accuracy']:.4f} (Accuracy)
   - 최종데이터_균형.xlsx: {best_balanced['Accuracy']:.4f} (Accuracy)
   - 차이: {best_imbalanced['Accuracy'] - best_balanced['Accuracy']:.4f}

2. 모델별 성능 차이
   - 불균형 데이터: SVM이 최고 성능
   - 균형 데이터: Naive Bayes가 최고 성능

3. F1-Macro vs F1-Weighted
   - 불균형 데이터: F1-Weighted가 F1-Macro보다 높음 (클래스 불균형 영향)
   - 균형 데이터: F1-Macro와 F1-Weighted가 유사함

📁 저장된 파일들:
- models/최종데이터_SVM_model.pkl
- models/최종데이터_균형_Naive Bayes_model.pkl
- results/최종데이터_performance_summary.csv
- results/최종데이터_균형_performance_summary.csv
- results/*_confusion_matrix.png

📊 결론:
- 한국어 감정 분류의 베이스라인 성능을 확인했습니다.
- N-gram 기반 전통적 머신러닝의 한계를 보여줍니다.
- 더 정확한 성능을 위해서는 딥러닝 모델이 필요합니다.
- 데이터 불균형이 성능에 미치는 영향을 확인했습니다.
"""

# 보고서 저장
with open('종합_베이스라인_성능결과.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ 종합 베이스라인 성능 결과 보고서가 생성되었습니다!")
print("📁 파일명: 종합_베이스라인_성능결과.txt")
print("\n" + "="*60)
print(report)
