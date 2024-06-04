import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'

# 이후로는 그래프에 한글이 제대로 표시됩니다.
# 예제: 간단한 그래프 그리기
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('테스트 그래프')
plt.xlabel('x 축')
plt.ylabel('y 축')
plt.show()
