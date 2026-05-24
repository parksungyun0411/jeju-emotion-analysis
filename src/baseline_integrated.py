import pandas as pd
from datetime import datetime

# 두 데이터셋의 성능 결과를 읽어오기
df_imbalanced = pd.read_csv('results/최종데이터_performance_summary.csv')
df_balanced = pd.read_csv('results/최종데이터_균형_performance_summary.csv')

# 통합 보고서 생성
report = f"""🎯 N-gram 기반 감정 분류 베이스라인 성능 평가 통합 결과
============================================================

📊 실험 정보:
  - 실험 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  - 벡터화: TF-IDF (1-3 gram, max_features=30,000)
  - 분할: 80% 학습, 20% 테스트
  - 평가 지표: Accuracy, F1-Macro, F1-Weighted
  - 모델: Logistic Regression, SVM, Random Forest, Naive Bayes

📈 데이터셋별 상세 성능 결과
============================================================

🔸 최종데이터.xlsx (불균형 데이터 - 72,879개, 테스트: 14,576개)
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

🔸 최종데이터_균형.xlsx (균형 데이터 - 12,950개, 테스트: 2,590개)
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

📊 모델별 성능 비교 (두 데이터셋 통합)
============================================================

🔸 Logistic Regression
------------------------------------------------------------
데이터셋              정확도        F1-Macro      F1-Weighted   
------------------------------------------------------------
최종데이터.xlsx       {df_imbalanced[df_imbalanced['Model']=='Logistic Regression']['Accuracy'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='Logistic Regression']['F1_Macro'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='Logistic Regression']['F1_Weighted'].iloc[0]:.4f}
최종데이터_균형.xlsx   {df_balanced[df_balanced['Model']=='Logistic Regression']['Accuracy'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='Logistic Regression']['F1_Macro'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='Logistic Regression']['F1_Weighted'].iloc[0]:.4f}

🔸 SVM
------------------------------------------------------------
데이터셋              정확도        F1-Macro      F1-Weighted   
------------------------------------------------------------
최종데이터.xlsx       {df_imbalanced[df_imbalanced['Model']=='SVM']['Accuracy'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='SVM']['F1_Macro'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='SVM']['F1_Weighted'].iloc[0]:.4f}
최종데이터_균형.xlsx   {df_balanced[df_balanced['Model']=='SVM']['Accuracy'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='SVM']['F1_Macro'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='SVM']['F1_Weighted'].iloc[0]:.4f}

🔸 Random Forest
------------------------------------------------------------
데이터셋              정확도        F1-Macro      F1-Weighted   
------------------------------------------------------------
최종데이터.xlsx       {df_imbalanced[df_imbalanced['Model']=='Random Forest']['Accuracy'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='Random Forest']['F1_Macro'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='Random Forest']['F1_Weighted'].iloc[0]:.4f}
최종데이터_균형.xlsx   {df_balanced[df_balanced['Model']=='Random Forest']['Accuracy'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='Random Forest']['F1_Macro'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='Random Forest']['F1_Weighted'].iloc[0]:.4f}

🔸 Naive Bayes
------------------------------------------------------------
데이터셋              정확도        F1-Macro      F1-Weighted   
------------------------------------------------------------
최종데이터.xlsx       {df_imbalanced[df_imbalanced['Model']=='Naive Bayes']['Accuracy'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='Naive Bayes']['F1_Macro'].iloc[0]:.4f}      {df_imbalanced[df_imbalanced['Model']=='Naive Bayes']['F1_Weighted'].iloc[0]:.4f}
최종데이터_균형.xlsx   {df_balanced[df_balanced['Model']=='Naive Bayes']['Accuracy'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='Naive Bayes']['F1_Macro'].iloc[0]:.4f}      {df_balanced[df_balanced['Model']=='Naive Bayes']['F1_Weighted'].iloc[0]:.4f}

📊 종합 분석 및 결론
============================================================

🔍 주요 발견사항:

1. 데이터 불균형의 영향
   - 불균형 데이터에서 전반적으로 더 높은 정확도 달성
   - 최종데이터.xlsx: 최고 52.54% (SVM)
   - 최종데이터_균형.xlsx: 최고 30.77% (Naive Bayes)
   - 차이: 21.77%p

2. 모델별 성능 특성
   - SVM: 불균형 데이터에서 최고 성능 (52.54%)
   - Naive Bayes: 균형 데이터에서 최고 성능 (30.77%)
   - Random Forest: 두 데이터셋 모두에서 상대적으로 낮은 성능
   - Logistic Regression: 중간 수준의 안정적 성능

3. F1-Score 분석
   - 불균형 데이터: F1-Weighted > F1-Macro (클래스 불균형 영향)
   - 균형 데이터: F1-Macro ≈ F1-Weighted (균형적 분포)

4. 모델별 성능 순위 (불균형 데이터)
   1위: SVM (52.54%)
   2위: Logistic Regression (52.33%)
   3위: Naive Bayes (50.39%)
   4위: Random Forest (49.64%)

5. 모델별 성능 순위 (균형 데이터)
   1위: Naive Bayes (30.77%)
   2위: Logistic Regression (30.19%)
   3위: SVM (29.50%)
   4위: Random Forest (26.68%)

📁 저장된 파일들:
- models/최종데이터_SVM_model.pkl
- models/최종데이터_균형_Naive Bayes_model.pkl
- results/최종데이터_performance_summary.csv
- results/최종데이터_균형_performance_summary.csv
- results/*_confusion_matrix.png

📊 결론:
- 한국어 감정 분류의 베이스라인 성능을 확인했습니다.
- N-gram 기반 전통적 머신러닝의 한계를 보여줍니다.
- 데이터 불균형이 성능에 미치는 영향을 확인했습니다.
- 더 정확한 성능을 위해서는 딥러닝 모델이 필요합니다.
- 모델 선택 시 데이터 특성(불균형/균형)을 고려해야 합니다.
"""

# 보고서 저장
with open('베이스라인_성능결과_통합.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ 통합 베이스라인 성능 결과 보고서가 생성되었습니다!")
print("📁 파일명: 베이스라인_성능결과_통합.txt")
print("\n" + "="*60)
print(report)
