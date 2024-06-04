import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 분석 요점 (카테고리)
categories = ['안타', '홈런', '타점', '득점', '타율', '루타']

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'

# 데이터 불러오기
data1 = pd.read_csv('15년부터 23년까지 타자.csv')
data2 = pd.read_csv('23년 전체 타자.csv')
data3 = pd.read_csv('24년 롯데 타자.csv')

# 각 데이터 집합에서 필요한 열만 선택하여 평균값 계산하기
data1_means = data1[categories].mean()
data2_means = data2[categories].mean()
data3_means = data3[categories].mean()

means_value = [data1_means, data2_means, data3_means]

# 데이터를 numpy 배열로 변환
data = np.array([data1_means, data2_means, data3_means])

# 각 데이터 집합의 색상과 레이블
colors = ['b', 'r', 'g']
labels = ['15년부터 23년까지 타자', '23년 전체 타자', '24년 롯데 타자']

# 레이더 차트 그리기 준비
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
data = np.concatenate((data, data[:,[0]]), axis=1)  # 데이터를 폐쇄형으로 만듦
angles += angles[:1]

# 레이더 차트 그리기
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for d, color, label in zip(data, colors, labels):
    ax.fill(angles, d, color=color, alpha=0.25)
    ax.plot(angles, d, color=color, linewidth=2, label=label)

# 축 라벨 설정
plt.xticks(angles[:-1], categories)

# 축 레이블 추가
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

#범례 표시
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
   
# 축 설정
ax.set_theta_offset(np.pi / 2)  # 시작점을 위로 조정
ax.set_theta_direction(-1)  # 시계 반대 방향으로 그리기

plt.xticks(angles[:-1], categories)  # 카테고리 라벨 설정

plt.legend()
plt.show()

