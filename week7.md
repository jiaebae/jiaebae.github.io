##### 활동 일자: 2024.08.13
#### 목표 : ResNet 코드 실습하기2, 양자화에 대해 알아보기

### ResNet 모델 정의하기
![](https://jiaebae.github.io/lib/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-08-13%20204051.png)

```
class ResNet(nn.Module):
	def __init__(self.num_classes=10):
		super(ResNet, self).__init__()
		
		# 기본 블록
		self.b1 = BasicBlock(in_channels=3, out_channels=64)
		self.b2 =  BasicBlock(in_channels=64, out_channels=128)
		self.b3 =  BasicBlock(in_channels=128, out_channels=256)
		
		# 풀링을 최댓값이 아닌 평균값으로
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
		
		# 분류기
		self.fc1 = nn.Linear(in_features=4096, out_features=2048)
		self.fc2 = nn.Linear(in_features=4096, out_features=2048)
		self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
		
		self.relu = nn.ReLu()
```
10개의 클래스를 갖도록 num_classes를 10으로 저장함.
super()를 이용하여 모듈의 요소들을 불러옴.
기본 블록을 3개 사용하기 때문에 3개 만들어 줌.
입력으로 채널 수가 3이 들어와서 최종적으로 채널 수 256이 나가는 형태가 됨.
풀링은 평균풀링을 사용함. 이미지의 크기를 반으로 줄이기 위해서 kernel_size=2, stride=2로 함.
stride는 커널의 이동거리임.
분류기는 3개의 MLP층을 가짐.
4096개의 특징을 받아서 최종적으로 10개 클래스에 대한 확률값을 내보냄.
활성화 함수로는 ReLu를 사용함.

ResNet의 순전파 정의
```
def forward(self, x):
	# 기본 블록과 풀링층 통과
	x = self.b1(x)
	x = self.pool(x)
	x = self.b2(x)
	x = self.pool(x)
	x = self.b3(x)
	x = self.pool(x)
	
	# 분류기의 입력으로 사용하기 위한 평탄화
	x = torch.flatten(x, start_dim=1)
	
	# 분류기로 예측값 출력
	x = self.fc1(x)
	x = self.relu(x)
	x = self.fc2(x)
	x = self.relu(x)
	x = self.fc3(x)
	
	return x
```
첫번째 블럭 지나고 풀링 해주고, 두번째 블럭 지나고 풀링 해주고, 세번째 블럭 지나고 풀링 해줌.
기본 블럭과 풀링층을 통과 했을 때까지는 2차원 이미지의 형태임.
출력값은 1차원 벡터이기 때문에 평탄화 작업을 해줌.
평탄화 한 값을 분류기에 넣어 최종적인 예측값을 출력함.

### 모델 학습하기
![](https://jiaebae.github.io/lib/media/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-08-14%20175116.png)

데이터 전처리 정의
```
import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torchvision.utils.data.datloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose[
	RandomCrop((32,32), padding=4), # 랜덤 크롭핑
	RandomHorizontalFlip(p=0.5), # 랜덤 y축 대칭
	ToTensor(),
	Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
]
```
Compose를 통해 전처리를 모아줌.
랜덤 크롭핑을 통해 필요한 부분(32x32)만 보고 세로 4개에 픽셀을 0으로 채움.
랜덤하게 y축 대칭을 해주고 이미지를 텐서로 바꿔줌.
Normalize()를 통해 이미지를 정규화해줌.

데이터불러오기
```
# 데이터셋 정의
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

# 데이터로더 정의
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```

모델 정의하기
```
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet(num_classes=10)
model.to(device)
```
device는 gpu면 cuda를 사용하고 아니면 cpu를 사용함.
모델은 ResNet이고 클래스는 10개를 사용함.
모델을 device로 보냄.

학습 루프 정의
```
lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
	iterator = tqdm.tqdm(train_loader)
	for data, label in iterator:
	
		# 최적화를 위해 기울기를 초기화
		optim.zero_grad()
		
		# 모델의 예측값
		preds = model(data.to(device))
		
		# 손실 계산 및 역전파
		loss = nn.CrossEntropyLoss()(preds, label.to(device))
		loss.backward()
		optim.step()
		iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
torch.save(model.state_dict(), "ResNet.pth")
```
학습율(lr) 1e-4 = 10^-4임.
Adam최적화 사용하여 모델의 파라미터를 최적화함. 확률은 미리 정의 했던 학습율이 적용됨.
for문을 통해 총 30번 학습 반복함.
데이터 로더의 프로그래스 바를 위한 tqdm으로 감싸줌.
데이터와 정답을 받아 온 다음에 기울기 초기화 해줌.
모델 예측값 출력한 다음 손실 계산 해주고 오차 역전파하고 최적화 진행해 준 다음 epoch마다 손실 출력 해줌.
학습이 완료된 가중치를 torch.save()를 이용해서 ResNet.pth라는 이름으로 저장함.

### 모델 성능 평가
ResNet 성능 확인해보기
```
model.load_state_dict(torch.load("ResNet.pth"), map_location=device)

num_corr = 0

with torch.no_grad():
	for data, label in test_loader:
	
		output = model(data.to(device))
		preds = output.data.max(1)[1]
		corr = preds.eq(label.to(device).data).sum().item()
		num_corr += corr
		
	print(f"Accuracy:{num_corr/leln(test_data)}")
```
"ResNet.pth"를 torch.load를 이용하여 device라는 위치에 불러옴.
전체 맞춘 갯수(num_corr) 0으로 초기화.
기울기 사용하지 않으니 no_grad로 선언.
test_loader로 data와 label 받아온 다음 모델 예측값 내보내 주고, output값의 인덱스만 받아오고 인덱스가 정답과 일치하는지 확인하고 일치한다면 num_corr값 갱신해주고 최종적으로 분류 정확도를 출력해줌.

### Quantization(양자화)란?
Quantization은 실수형 변수(floating-point type)를 정수형 변수(integer or fixed point)로 변환하는 과정을 뜻함.
- Quantization은 weight나 activation function의 값이 어느 정도의 범위 안에 있다는 것을 가정하여 이루어지는 모델 경량화 방법임
		- > floating point로 학습한 모델의 weight 값이 -10 ~ 30 의 범위에 있다는 경우의 예시에 대해, 최소값인 -10을 uint8의 0에 대응시키고 30을 uint8의 최대값인 255에 대응시켜서 사용한다면 32bit 자료형이 8bit 자료형으로 줄어들기 때문에 전체 메모리 사용량 및 수행 속도가 감소하는 효과를 얻을 수 있음
    
- 이와 같이 Quantization을 통하여 효과적인 모델 최적화를 할 수 있는데, float 타입을 int형으로 줄이면서 용량을 줄일 수 있고 bit 수를 줄임으로써 계산 복잡도도 줄일 수 있음
    
- 또한 정수형이 하드웨어가 연산하기에 더 간편하다는 장점이 있음
    
    > [정리] Listed advantages:
    > 
    > > ① 모델의 사이즈 축소
    > > 
    > > ② 모델의 연산량 감소
    > > 
    > > ③ 효율적인 하드웨어 사용