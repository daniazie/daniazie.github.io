---
layout: post
title: "[수학 공부] Why Machines Learn 1주차: The Perceptron"
published: true
date: 2025-09-09
math: true
categories: 
    - Study
    - 수학
tags: KHUDA
---

## 개요
KHUDA 자연언어처리 트랙 뒷타임 수학 세션은 Anil Ananthaswamy가 쓰신 Why Machines Learn: The Elegant Math Behind Modern AI라는 책을 다룹니다. 책 내용을 번역하여 정리하고 뒷타임 세션 때 강의해드립니다. 그리고 욕심이 엄청 높아져서 해당 책에서 공부하는 수학을 코드로 응용하는 것도 해보겠습니다.

## 내용 정리 - 교재 1장과 2장
### 지도학습

$$y = w_1x_1 + w_2x_2 + \cdots +w_nx_n = \sum_{i=1}^n w_ix_i$$


을 가정합시다. 수학적으로 지도학습은 각 가중치 $w_i$를 추출하도록 학습하는 목표입니다. 즉, 특징 (features)와 타겟값의 관계를 분석하는 것입니다.

${x, y}$를 가지는 데이터를 훈련데이터하고 하는데 labeled data 또는 annotated data라고도 합니다. 

<blockquote><dl><dt>노트!<br>
<dd>지연언어처리 논문에서 보통 annotated data라고 합니다</dd>
</dt></dl></blockquote>

### 첫 인공 뉴런
**McCulloch-Pitts (MCP) 뉴런:** 계산 단위인데, 계산만 가능하고 학습 불가능 &rarr; Rosenblatt's Perceptron

$$\begin{aligned}g(x) & = x_1 + x_2+\cdots+x_n \\ f(z) & =  \begin{cases} 0, z < \theta \\ 1, z \geq \theta \end{cases} \\ y = f(g(x)) & = \begin{cases} 0, g(x) < \theta \\ 1, g(x) \geq \theta \end{cases} \end{aligned}$$


### 실수로써 학습하기
#### 학습이란
* **인공 모델:** 데이터를 분석하면서 패턴을 추출하기
* **인간:** 한 뉴런의 아웃풋이 다른 뉴런을 트리거할 때는 뉴런 간의 연결을 강화하는 과정 &rarr; Hebbian Learning이라고 함 

&rarr; Rosenblatt은 인간의 뉴런과 같이 서로 트리거함으로써 공부하는 인공 뉴런을 발전하고자 하셨습니다.

#### Augmented MCP (퍼셉트론)
* 데이터에 없는 샘플의 값을 알 수 있도록 데이터 특징과 레이블 간 관계를 학습하기

$$\begin{align}g(x) & = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = \sum_{i=1}^{n}w_ix_i + b \\ f(z) & = \begin{cases}-&1, z \leq 0 \\ & 1, z > 0\end{cases} \\ y = f(g(x)) & = \begin{cases} - & 1, g(x) \leq 0 \\ & 1, g(x) > 0\end{cases}\end{align}$$

**벡터로 표현하려면**

$$y = \begin{cases} - & 1, \mathbf{w}^\top\mathbf{x} \leq 0 \\ & 1, \mathbf{w}^\top\mathbf{x} > 0 \end{cases}, \text{ where } \mathbf{w} = \{w_0, w_1, \cdots, w_n\} \text{ and } \mathbf{x} = \{1, x_1, \cdots, x_n\}
$$

**파라미터 업데이트**

$$\mathbf{w}_{\text{new}} = \mathbf{w}_\text{old} + y\mathbf{x} \text{ if } y\mathbf{w}^\top\mathbf{x} \leq 0$$

> **왜 $y\mathbf{w}^\top\mathbf{x}$인가?**<br>
$y$과 $\mathbf{w}^\top\mathbf{x}$가, 즉 실제값과 예측값이, 똑같으면 스칼라곱은 양수고, 예측값이 틀리면 음수가 나와서 잘 예측한지 확인하도록 스칼라곱 과정을 진행합니다.<br> <br>
&rarr; 즉, 데이터와 적합하는 가중치와 편향을 알아보는 목표입니다

| MCP | Perceptron | 
| ----| -----|
|인풋은 이진값이어야 함 | 인풋은 이진값이 아닌 값을 가져도 됨|
|가중치 없음 | 가중치 有 |
|아웃풋은 0이나 1임| 아웃풋은 -1이나 1임|
|학습 불가 |가중치와 편향을 학습 가능|


## 파이썬으로 응용하기
자... 위 개념을 기반으로 퍼셉트론 직접 만들어봅시다. 

> 참고: [Colab](https://colab.research.google.com/drive/1GdTKzX7ulZOretLsK4IqmEVaf-ku3m5O?usp=sharing)

해당 실습은 numpy를 이용합니다.


```python
import numpy as np
```

우리는 Perceptron이라는 파이썬 클래스를 만들고 클래스 속 함수를 정의합니다.

```python
class Perceptron:
    def __init__(
            self, 
            inputs, 
            labels, 
            epoch,
        #    learning_rate = 0.05
            ):

        """
        클래스 초기화하는 함수
        
        self를 통해 '이 클래스가 원래 요 변수들을 갖는다'라고 정할 수 있습니다. 
        해당 코드는 bias를 w[0]로 정의하여, 인풋 배열에 x[0]=1 산입합니다  (즉, [1, x1, x2])

        그리고, 교재에서 learning rate 아직도 안 나타났지만, learning rate 어떻게 적용할지를 아실  수 있도록 주석으로 learning rate에 해당 코드를 놓았습니다.
        """

        self.inputs = inputs
        self.inputs = np.array([np.insert(inp, 0, 1) for inp in inputs])
        self.weights = np.zeros(len(self.inputs[0]))
        self.labels = labels
        self.epoch = epoch
        # self.lr = learning_rate
        
    def step_function(self, product):
        """
        이진 분류 실행하는 계단함수 코드
        """
        if product <= 0:
            return -1
        else:
            return 1

    def update(self, weights, inputs, labels):
        """
        가중치 업데이트
        """
        return weights + labels*inputs

    def train(self):
        """
        훈련 코드

        epoch마다 각 샘플에 대한 y'를 계산하고, 필요하면 파라미터 업데이트 실행합니다.
        """
        epoch = self.epoch

        for e in range(epoch):
            for i in tqdm(range(len(self.inputs)), desc=f'Epoch {e+1}'):
                pred = self.weights @ self.inputs[i]
                acc = self.step_function(pred) * self.labels[i]
                if acc <= 0:
                    self.weights = self.update(self.weights, self.inputs[i], self.labels[i])
                #   self.weights = self.weights + self.lr * (self.labels[i] * self.inputs[i])
    
    def predict(self, inputs):
        """
        예측 코드

        추출된 가중치를 통해 실험 셋의 레이블을 계산합니다.
        """
        weights = self.weights
        inputs = np.array([np.insert(inp, 0, 1) for inp in inputs])
        product = np.array([weights @ x for x in inputs])
        pred = np.array([self.step_function(pred) for pred in product])

        return pred
```

### 보너스: 기초 평가 방법들

나중 6주차 때 평가에 대해 공부할 테지만, 그래도 자연언어처리에 사용하는 평가의 기초 공식을 간략하게 공부하면 좋지 않을까 싶었습니다.

#### Accuracy
제일 기분 공식은 정확성(Accuracy)라고, 전체 샘플에 비해 모델을 어느 정도 잘 맞추는지를 알려주는 공식입니다. 즉, 

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$$

코드로:
```python
def evaluate(pred_labels, label):
    count = 0
    for pred, true in zip(pred_labels, label):
        if pred == true:
            count += 1
    return count/len(pred_labels)
```

하지만, 클래스 불균형이 있으면 단순히 Accuracy를 이용하면 불확실한 결과도 나올 수도 있습니다.

따라서, confusion matrix를 이용하는 평가 공식을 이용합니다.

#### Le Confusion Matrix

예측값은 4가지로 분류될 수 있습니다. 첫번째 경우 true positive이라고 잘 예측된 positive한 레이블, 두번째는 true negative이라고 잘 예측된 negative한 레이블, 셋번째 false positive이라고 못 맞추는 negative한 레이블, 그리고 false negative이라고 못 맞추는 positive한 레이블. 즉,


|Pred/True|Positive|Negative
|------|------|------|
**Positive**|TP|FP|
**Negative**|FN|TN|

따라서, confusion matrix를 이용해 정확성을 다시 정의하려면,


$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

즉, 단순히 모델은 어느 정도 잘 맞추느냐를 나타내는 공식입니다.

#### 관련성 (Relevance)

관련성은 모델이 얼마 정도로 우리가 원하는 정보를 추출하는지를 알려주는 평가방법입니다. 관련성 기반 평가 계산은 크게 3가지 있습니다.

**Precision**

Precision은 모델이 positive하는 것으로 예측하는 샘플에 비해 올바르게 positive하는 것을 예측하는 값이 나타납니다. 즉,


$$\text{Precision} = \frac{TP}{\text{TP} + \text{FP}}$$

**Recall**

Recall은 실제 positive 값 샘플에 비해 모델이 올바르게 예측하는 positive값 얼마인지를 나타내는 값입니다. 즉,


$$\text{Recall} = \frac{TP}{\text{TP} + \text{FN}}$$

**Precision vs. Recall**

Precision과 Recall 중에 어떤 걸로 사용하면 더 나을까요? 정답은 케이스바이케이스입니다. 

FP 줄이도록 하고자 하면 Precision 이용하면 더 낫고 FN 줄이도록 하고자 하면 Recall 이용하면 더 좋죠. 예를 들어 암 검사에서 FP 나타나면 환자에게 비용이 많아지며 FP를 줄도록 평가하면 더 좋고, 피슁이 발생하면 정말 大 일 나겠어 Recall 위주로 평가하면 제일 베스트죠. 

**$F_1$ 점수**

$F_1$ 점수는 Precision과 Recall 같이 계산하는 평가방법인데, $\beta=1$인 $F_\beta$ 점수 나타납니다. $\beta$는 가중치이므로 $\beta=1$이라면 Precision과 Recall을 대칭으로 대표해 계산합니다. 즉, Precision과 Recall 조화 평균(harmonic mean)을 나타냅니다.


$$F_\beta = \frac{(1+\beta^2)\cdot\text{precision}\cdot\text{recall}}{\beta^2\cdot\text{precision}+\text{recall}}$$

$\beta > 1$라면, $w_\text{recall} > w_\text{precision}$이라고 생각해도 되고 $\beta < 1$ 경우 $w_\text{recall} < w_\text{precision}$이라고 생각해도 됩니다. 즉, $\beta =1$일 때,  $w_\text{recall} = w_\text{precision}$입니다.


$$\begin{align*} \beta & = 1, \\ F_1 & = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision} + \text{recall}} = \frac{2\text{TP}}{2\text{TP}+\text{FP}+\text{FN}}\end{align*}$$

> 노트!
LLM 평가할 때 $F_1$ 위주로 이용합니다.