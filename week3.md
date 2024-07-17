##### 활동 일자: 2024.07.16
#### 목표 : 문채운 선배님의 레포 분석하기 - resnet

# ResNet(Residual Network)란?
- 딥러닝에서 널리 사용되는 인공신경망 구조
- 2015년 Microsoft의 연구팀이 개발
- Residual Connection을 사용하여 심층 신경망에서 발생하는 문제를 해결
- 사용되는 곳: 이미지 분류, 객체 검출 등.
-일반적으로 신경망이 깊어질수록 더 많은 특징을 학습할 수 있지만, 너무 깊어지면 학습이 어려워지고 성능이 떨어질 수 있음. 이를 "기울기 소실(vanishing gradient)" 문제라고 함. ResNet은 이러한 문제를 해결하기 위해 Residual Connection을 도입함.

### Residual Connection
- 입력값을 출력값에 더하는 구조
- 신경망의 층(layer)을 거친 결과에 원래 입력을 더해줌
- y = F(x)+x
	-x: 입력값
	-F(x): 여러 신경망 층을 거친 출력값
	-y: 최종 출력값
	이 구조 덕분에 신경망이 깊어지더라도 학습이 잘 진행됨.
	입력값을 그대로 출력에 더하기 때문에, 필요한 경우 모델이 단순히 항등 함수를 학습할 수도 있음.
	이는 깊은 네트워크에서 중요한 역할을 함.

### Residual Block
- Residual Connection을 포함한 더 큰 구조
- ResNet의 기본 구성 단위
- 형태
	1. 입력 x
	2. 두 개의 신경망 층
	3. 출력값 F(x)
	4. Residual Connection을 통해 최종 출력값 y = F(x)+x
	이 블록들이 여러 개 모여 하나의 ResNet이 됨.

### ResNet의 장점
1. 깊은 네트워크 학습 가능: Residual Connection 덕분에 깊은 네트워크도 효과적으로 학습할 수 있음.
2. 성능 향상: 이미지 분류와 같은 다양한 작업에서 뛰어난 성능을 보임.
3. 효율적 구현: 비교적 간단한 구조로도 높은 성능을 발휘함.

### 선배 코드에서의 ResNet
```
def get_resnet152():
    model_id = "Wespeaker/wespeaker-voxceleb-resnet152-LM"
    model_name = model_id.replace("Wespeaker/wespeaker-", "").replace("-", "_")

    root_dir = hf_hub_download(model_id, filename=model_name+".onnx").replace(model_name+".onnx", "")

    import os
    if not os.path.isfile(root_dir+"avg_model.pt"):
        os.rename(hf_hub_download(model_id, filename=model_name+".pt"), root_dir+"avg_model.pt")
    if not os.path.isfile(root_dir+"config.yaml"):
        os.rename(hf_hub_download(model_id, filename=model_name+".yaml"), root_dir+"config.yaml")

    resnet = wespeaker.load_model_local(root_dir)

    #print("Compile model for the NPU")
    #resnet.model = intel_npu_acceleration_library.compile(resnet.model)

    def resnet152(ado, sample_rate=None):
        if isinstance(ado, str):
            return resnet.recognize(ado)
        else:
            return recognize(resnet, ado, sample_rate)

    resnet152.__dict__['register'] = lambda *args, **kwargs: resnet.register(*args, **kwargs)

    return resnet152
```

분석
```
def get_resnet152():
    model_id = "Wespeaker/wespeaker-voxceleb-resnet152-LM"
    model_name = model_id.replace("Wespeaker/wespeaker-", "").replace("-", "_")
```
get_resnet152라는 이름의 함수 정의
'model_id'에 Wespeaker/wespeaker-voxceleb-resnet152-LM을 저장
'model_name'은'model_id'에서 "Wespeaker/wespeaker-"를 제거하고, 하이픈을 언더스코어로 대체하여 생성된 모델 이름. 따라서 voxceleb_resnet152_LM이 모델 이름임.

```
root_dir = hf_hub_download(model_id, filename=model_name+".onnx").replace(model_name+".onnx", "")Your Code
```
'hf_hub_download' 함수는 'model_id'와 파일 이름을 사용하여 모델과 파일을 다운로드함.
다운로드한 파일을 root_dir변수에 저장함.

```
import os
    if not os.path.isfile(root_dir+"avg_model.pt"):
        os.rename(hf_hub_download(model_id, filename=model_name+".pt"), root_dir+"avg_model.pt")
    if not os.path.isfile(root_dir+"config.yaml"):
        os.rename(hf_hub_download(model_id, filename=model_name+".yaml"), root_dir+"config.yaml")
```
os모듈을 임포트함.
avg_model.pt 파일이 root_dir에 없는 경우, '.pt' 파일을 다운로드하여 'avg_model.pt'로 이름을 변경함.
config.yaml파일이 root_dir에 없는 경우, '.yaml'파일을 다운로드하여 'config.yaml'로 이름을 변경함.

```
resnet = wespeaker.load_model_local(root_dir)
```
wespeaker 모듈의 load_madel_local함수를 사용하여 root_dir에서 모델을 로드함.

```
	#print("Compile model for the NPU")
	#resnet.model = intel_npu_acceleration_library.compile(resnet.model)
```
주석 처리되어 있으며, 주석을 해제하면 모델을 NPU(Neural Processing Unit)를 위해 컴파일함.
NPU는 딥러닝 알고리즘 연잔에 최적화된 프로세서임.


```
def resnet152(ado, sample_rate=None):
        if isinstance(ado, str):
            return resnet.recognize(ado)
        else:
            return recognize(resnet, ado, sample_rate)
```
resnet152라는 이름의 내부 함수를 정의함.
입력 ado가 문자열인 경우, resnet 객체의 recognize함수를 호출함.
그렇지 않은 경우, *recognize 함수*를 호출함.
- **recognize함수**:음성데이터에서 임베딩 벡터를 추출하고, 모델의 임베딩 테이블과 비교하여 가장 유사한 항목을 찾음.
```
def recognize(model, pcm, sample_rate):
    q = extract_embedding(model, pcm, sample_rate)
    best_score = 0.0
    best_name = ''
    for name, e in model.table.items():
        score = model.cosine_similarity(q, e)
        if best_score < score:
            best_score = score
            best_name = name
        del score
        gc.collect()
    return {'name': best_name, 'confidence': best_score}
```
recognize라는 이름의 함수를 정의 매개변수로 model(음성 인식 모델 객체),pcm(음성 데이터),sample_rate(샘플링 레이트)를 받음
extract_embedding함수는 pcm과 sample_rate를 사용하여 주어진 모델에서 임베딩 벡터를 추출하여 q에 저장함.
best_score는 현재까지 발견된 최고의 유사도 점수를 저장함.
best_name은 현재까지 발견된 최고의 유사도를 가지는 항목의 이름을 저장함.
model.table은 모델이 가지고 있는 임베딩 테이블임. 이 테이블은 name과 e의 쌍으로 이루어져 있음.
e와 q간의 코사인 유사도 점수를 계산함.
계산된 점수가 현재 best_score보다 높으면, best_score와 best_name을 업데이트함.
del score와 gc.collect()는 메모리 관리를 위해 사용됨. 점수를 삭제하고 Garbage Collector를 호출하여 메모리를 해제함.
최종적으로 가장 높은 유사도를 가지는 best_name과 best_score를 딕셔너리 형태로 반환.


```
resnet152.__dict__['register'] = lambda *args, **kwargs: resnet.register(*args, **kwargs)
```
resnet152함수 객체에 register메서드를 추가함. 이는 lambda함수로 resnet객체의 register메서드를 호출함. 함수의 기능은 새로운 데이터를 등록을 함.

```
return resnet152
```
최종적으로 resnet152함수를 반환함.