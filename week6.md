##### 활동 일자: 2024.08.06
#### 목표 : ResNet 코드 실습하기

## 기본 블록 정의하기

![]

ResNet 기본블록
```
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
def __init__(self, in_channels, out_channels, kernel_size=3):
	super(BasicBlock, self).__init__()
	
	#합성곱층 정의
	self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kerner_size, padding=1)
	self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kerner_size, padding=1)
	self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
	
	#배치 정규화층 정의
	self.bn1 = nn.BatchNorm2d(num_features=out_channels)
	self.bn2 = nn.BatchNorm2d(num_features=out_channels)
	
	self.relu = nn.ReLU()
```

```
import torch
import torch.nn as nn
```
파이토치와 신경망 불러와 줌.

```
class BasicBlock(nn.Module):
def __init__(self, in_channels, out_channels, kernel_size=3):
	super(BasicBlock, self).__init__()
```
BasicBlock이라는 이름으로 클래스 만들어 줌.
초기화 함수를 정의 해 줌. 'in_channels'은 입력 채널 수, 'out_channels'는 출력 채널 수, 'kernel_size'는 합성곱이 갖는 커널의 크기임.
super()를 이용해 nn.Module을 가져옴.

```
#합성곱층 정의
	self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kerner_size, padding=1)
	self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kerner_size, padding=1)
	self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```
2개의 합성곱층과 다운샘플층을 정의해 줌.

```
#배치 정규화층 정의
	self.bn1 = nn.BatchNorm2d(num_features=out_channels)
	self.bn2 = nn.BatchNorm2d(num_features=out_channels)
	self.relu = nn.ReLU()
```
합성곱을 한 번 거칠 때마다 배치 정규화도 필요하기 때문에 배치 정규화 2개 정의해 줌.

```
self.relu = nn.ReLU()
```
활성화 함수 정