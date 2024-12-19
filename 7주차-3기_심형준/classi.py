import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns
import joblib

# 1. CSV 파일 불러오기
df1 = pd.read_csv('./3rd_assignment/7주차-3기_심형준/sit.csv')  # 'your_data.csv'는 여러분의 CSV 파일 경로
df2 = pd.read_csv('./3rd_assignment/7주차-3기_심형준/stand.csv')

df = pd.concat([df1, df2], ignore_index=True)

# 2. 데이터 확인
print("데이터 확인:\n", df.head())

# 3. 'Image' 열과 'Label' 열을 제외한 특성 데이터를 준비
X = df.drop(['Image', 'Label'], axis=1)  # 'Image'와 'Label' 열을 제외한 특성 데이터
y = df['Label']  # 'Label' 열을 레이블로 설정

# 5. 'Label'을 숫자로 변환 (이진 분류: 'sit' -> 0, 'stand' -> 1)
y = y.map({'sit': 0, 'stand': 1})

# 4. 훈련 데이터와 테스트 데이터로 분리 (stratify=y로 클래스 비율을 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 훈련 데이터에서 클래스 분포 확인
print(f"훈련 데이터 클래스 분포: \n{y_train.value_counts()}")

# 5. 데이터 표준화 (StandardScaler를 사용하여 특성 데이터를 표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 랜덤 포레스트 모델 학습
rf_model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 7. 예측
y_pred = rf_model.predict(X_test_scaled)

# 8. 성능 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)

'''
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.show()
'''