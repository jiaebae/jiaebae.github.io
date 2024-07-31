##### 활동 일자: 2024.07.30
#### 목표 : resnet구조를 통해 resnet은 왜 잘 작동하는지 알아보기

## Deepper bottleneck architecture
50층 이상의 깊은 모델에서는 Inception에서와 마찬가지로, 연산상의 이점을 위해 "bottleneck" layer(1x1 convolution)을 이용함.

![다운로드]()

기존의 Residual Block은 한 블록에 Convolution Layer(3X3) 2개가 있는 구조였음. Bottleneck 구조는 오른쪽 그림의 구조로 바꾸었는데 층이 하나 더 생겼지만 Convolution Layer(1X1) 2개를 사용하기 때문에 파라미터 수가 감소하여 연산량이 줄어들었음. 또한 Layer가 많아짐에 따라 Activation Function이 증가하여 더 많은 non-linearity가 들어감. 즉 Input을 기존보다 다양하게 가공할 수 있게 되었음.

*결론적으로 ResNet은 Skip Connection을 이용한 Shortcut과 Bottleneck 구조를 이용하여 더 깊게 층을 쌓을 수 있었음.*

ResNet이 잘 되는 이유
우리는 쉽게 Optimal depth를 알 수가 없음.
20층이 Optimal인지, 30층이 Optimal인지, 100층이 optimal인지 아무도 모름. 하지만 * degradation problem은 야속하게도 우리는 알 수 없는 optimal depth를 넘어가면 바로 일어남.

ResNet은 엄청나게 깊은 네트워크를 만들어주고, Optimal depth에서의 값을 바로 Output으로 보내버릴 수 있는데 가능한 이유는 **Skip connection** 때문임.
ResNet은 Skip connection이 존재하기 때문에 Main path에서 Optimal depth 이후의 Weight와 Bias가 전부 0에 수렴하도록 학습된다면 Optimal depth에서의 Output이 바로 Classification으로 넘어갈 수 있음. 즉, Optimal depth이후의 block은 모두 빈깡통과 같음.

예를 들어 27층이 Optimal depth인데 ResNet 50에서 학습을 한다면, 28층부터 Classification 전까지의 weight와 bias를 전부 0으로 만들어버림. 그러면 27층에서의 output이 바로 Classification에서 이용되고, 이는 Optimal depth의 네트워크를 그대로 사용하는 것과 같다고 볼 수 있음.

* * Degradation Problem: 딥러닝 모델의 레이어가 깊어졌을 때 모델이 수렴했음에도 불구하고 오히려 레이어 개수가 적을 때보다 모델의 trainig/test error가 더 커지는 현상이 발생. 이것은 오버피팅 때문이 아니라 네트워크 구조상 레이어를 깊이 쌓았을 때 최적화가 잘 안되기 때문에 발생하는 문제임.