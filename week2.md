##### 활동 일자: 2024.07.09
#### 목표 : 문채운 선배님의 레포 분석하기 - Whisper

# Whisper란?
- OpenAI에서 개발한 자동 음성 인식 모델(Automatic Speech Recognition,ARS).
- 웹에서 수집한 680,000시간 분량의 다국어 및 멀티태스크 감독 데이터를 기반으로 훈련된 자동 음성 인식(ARS) 시스템.
- whisper의 구조는 encoder-decoder transformer로 구현된 간단한 End to End 방식.
- zero-shot 성능이 좋음.

# 선배 코드에서의 Whisper

*whisper를 통해 음성을 텍스트로 변환하는 함수*
```
def get_whisper():
    model_size = "medium"  #@param ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    compute_type = "int8"  #@param ['float16', 'int8']

    return WhisperModel(model_size, device=DEVICE, cpu_threads=12, compute_type=compute_type).transcribe
```

*코드 하나씩 뜯어보기*

1. 모델 크기 설정('model_size'):
```
model_size = "medium"
```
	'model_size' 변수는 사용할 Whisper 모델의 크기를 지정함.
	Whisper 모델은 여러 크기로 제공되며, 작은 크기일수록 더 빠르지만 정확도가 낮을 수 있고,
	큰 크기일수록 더 느리지만 더 높은 정확도를 제공함.
2. 계산 유형 설정('compute_type'):
```
compute_type = "int8"
```
	'compute_type' 변수는 모델이 사용하는 숫자 형식을 지정함.
	float16 과 int8 중에서 선택할 수 있으며, int8은 더 낮은 정밀도를 가지지만
	메모리 사용량과 계산량이 적어 성능 향상에 도움이 될 수 있음.
3. Whisper 모델 초기화:
```
WhisperModel(model_size, device=DEVICE, cpu_threads=12, compute_type=compute_type)
```
	'WhisperModel' 객체를 초기화함. 이 객체는 다음의 인자를 받음.
	- 'model_size': 모델 크기. 이 코드에서는 medium.
	- 'device': 모델이 실행될 장치 (예; CPU or GPU). 이 코드에서 DEVICE는 cuda로 정의 되어있음(GPU).
	- 'cpu_threads': 모델이 CPU에서 실행될 경우 사용할 thread 수. 이 코드에서는 12개.
	- 'compute_type': 계산 유형. 이 코드에서는 int8.
4. 음성 인식 기능 반환:
```
return WhisperModel(...).transcribe
```
	초기화된 'WhisperModel' 객체의 'transcribe' 메서드를 반환함.
	transcirbe 메서드는 음성파일을 받아서 텍스트로 변환하는 기능을 수행함.
따라서 이 코드는 음성을 텍스트로 변환하는 Whisper 모델을 설정하고, 해당 모델의 음성 인식 기능을 반환하는 함수이다.