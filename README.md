## 워크샵 사전 준비 사항

에벤트 엔진, 세이지 메이커 노트북 설치, 소스 다운로드를 위해서 아래를 클릭해서 진행 해주세요.
- **사전 준비 사항**: [여기](prerequisite/Prerequisite.md)
    - 위의 링크를 누르시고 가이드를 따라 하신 후에 마지막에 ```git clone https://github.com/gonsoomoon-ml/churn-prediction-workshop2``` 를 실행하여 Git 소스를 다운로드 해주세요.


# 사용자 정의 컨테이너(BYOC) 및 Inference Pipeline을 이용한 고객 이탈 예측 모델 및 평가 
## (Churn Prediction Model-Inference-Pipeline-BYOC)

이 워크샵은 이전에 작성한 [고객 이탈 예측 모델 및 평가 (Churn Prediction Model)](https://github.com/gonsoomoon-ml/churn-prediction-workshop) 의 확장 버전 입니다. 주요한 차이점은 기존의 "고객 이탈 예측 모델 및 평가" 가 3개의 모델을 연속적으로 결합하여 Inference Pipeline을 만들었다고 하면, **이 워크샵은 Custom Model (Bring Your Own Container)을 추가하여 4개의 모델을 연속적으로 결합하여 Inference Pipeline을 만들었습니다.** 데이터는 기존과 같게 실제 미국 이동 통신 회사의 고객 데이타를 가지고 모델 학습 및 평가를 했습니다.

[정의] inference pipeline: 추론 파이프라인은 데이터에 대해 추론하기 위한 요청을 처리하는 2~5개의 컨테이너로 이루어진 선형 시퀀스로 구성된 Amazon SageMaker 모델입니다.(An inference pipeline is an Amazon SageMaker model that is composed of a linear sequence of two to five containers that process requests for inferences on data.)<br>
[개발자 가이드 정의 :inference pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html)


## 비즈니스 배경 (Business Background)

고객을 잃는 다는 것은 어느 기업이나 손실이 큽니다. **불만이 있는 고객을 미리 인지한다는 것은, 미리 이탈 방지 할 수 있는 여러 프로모션을 줄 수 있는 기회를 제공** 합니다. 머신 러닝은 데이타를 통해서 이런 이탈 고객을 예상할 수 있는 방법을 제공 합니다. 물론 머신 러닝이 100% 정확한 예상을 할 수는 없지만, 이탈에 영향을 주는 의미 있는 데이타를 확보할 경우에는 예측 정확도는 상당히 올라 갑니다. 이런 모델을 통해서 고객 이탈 방지할 수 있는 가치를 제공 할 수 있습니다.

## 기술적 내용 (Technical Coverage)

### 1. 데이타 플로우 관점의 Inference Pipeline

아래 그림은 input으로 Raw Input 이 SageMaker Endpoint에 제공되면, Inference Pipeline 의 4개의 컨테이너안에 있는 모델들이 각각 결과를 제공하는 과정을 보여 주고 있습니다. Raw Input이 첫 번재 컨테이너를 통과하면 전처리된 피쳐가 생성이 되고, 두 번째 컨테이너를 통과하면 차원이 축소된 피쳐가 생성이 되고, 세 번째 컨테이너를 통과 하면 score가 생성이 되고, 마지막 컨테이너를 통과 하면 True or False 값을 생성 합니다. 

![Fig.4.1.inference-pipeline-4-models](img/Fig.4.1.inference-pipeline-4-models.png)

### 2. 컨테이너간의 Input/Output Type 관점의 Inference Pipeline

아래 그림은 input으로 첫번째 컨테이너에 content-type:text/csv, data-type:string 이 입력으로 제공이 됩니다. 이후 첫번째 컨테이너 모델에서 결과를 받고 accept:text/csv, data-type:NumPy array를 output 으로 제공 합니다. 이후에 input으로 두번째 컨테이너에 content-type:text/csv, data-type:bytes가 입력으로 제공이 됩니다. 이후 두번째 컨테이너는 모델에서 결과를 받고 accept:text/csv, data-type:NumPy array를 output 으로 제공 합니다. 첫번째 컨테이너의 output인 data-type:NumPy array가 두 번째 컨테이너의 바로 가지 않고 data-type:bytes 로 바뀐 부분은 endpoint 내부에서 바꾸는 것으로 생각됩니다. (확인 필요 합니다.). 이후 세번째, 네번째 컨테이너도 같은 방식으로 보시면 됩니다.

![Fig.0.inference-pipeline-4-model-type](img/Fig.0.inference-pipeline-4-model-type.png)

### 3. Endpoint 에 추론 요청시에 4개의 컨테이너의 내부 동작

아래 그림은 Endpoint에 추론 요청시에 4개의 컨테이너의 각각에서 input_fn, predict_fn, output_fn 이 호출되는 것을 보여주고 있습니다. 아래 그림은 CloudWatch 에서 Endpoint 로그를 확인하면서 작성을 하였습니다. 개발시에 로그를 확인하면서 디버깅할 수 있습니다.

![Fig.0.Call-Endpoint-Flow](img/Fig.0.Call-Endpoint-Flow.png)

### 4. 워크샵 노트북 설명

- 1.1.Prepare-Churn-Data.ipynb
    - 고객 이탈 데이타를 다운로드 받고, 데이타를 train, validation, test로 나눕니다.
    
    
- 2.1.Write-Preprocess-Script.ipynb
    - 전처리와 후처리를 하는 알고리즘 스크립트 작성을 합니다. Custom Feature Transformer Model(BYOS) 및 Custom Post-Processing Model(BYOS) 인 두개의 모델 학습을 위한 스크립트 입니다. 이 스크립트는 학습/추론에 모두 사용 됩니다.


- 2.2.Train-Preprocess-Model.ipynb
    - 전처리 모델을 학습한 후에, train, validation 데이타를 전처리 모델에 입력으로 제공하고 전처리된 train, validation 데이터를 제공 받아 S3에 저장 합니다.


- 3.1.Write-PCA-Script.ipynb
    - Custom PCA Model(BYOC) 을 만들기 위한 스크립트 작성을 합니다. 이 스크립트는 학습/추론 모두에 사용됩니다.


- 3.2.Make-BYOC-Custom-PCA-Docker.ipynb
    - docker image를 생성하고, 이를 ECR(Elastic Container Registry)에 등록 합니다. **이 과정이 Custom하게 정의한 사용자의 알고리즘을 docker image로 만들어 ECR에 등록하여 사용하는 Bring Your Own Container(BYOC) 과정 입니다.**


- 3.3.Train-PCA-Model.ipynb
    - 위에서 생성한 custom docker image를 ECR에서 가져와서 학습을 하여 custom PCA model을 생성 합니다. 이 모델로 전처리된 train, validation 데이타를 모델에 입력하여 차원 축소된 train, validaion 데이터를 제공 받아 S3에 저장 합니다. 차원 축소는 69개의 피쳐를 25개로 축소를 하였습니다.


- 4.1.Train-XGBoost-Model.ipynb
    - Buil-in XGBoost docker image를 가져와서 이전에 생성한 차원 축소된 train, validation을 입력하여 학습 하여 모델을 생성 합니다.


- 5.1.Inference-Pipeline.ipynb
    - 기존 노트북에서 생성한 전처리 모델, Custom PCA 모델 , XGboost 모델의 모델 아티펙트, 학습/추론 docker image 및 환경 변수를 제공하여 1개의 단일 모델인 Inference Pipleline을 생성 합니다.


- 6.1.inference-RealtimeEndpoint.ipynb
    - 엔트포인트 컨피그에 위에서 생성한 inference pipeline model을 지정하고, 엔드포인트를 생성 합니다. 이후 실시간 추론을 하여 예측 결과를 확인 합니다.


- 6.2.inference_batch_transform.ipynb
    - transfomer를 생성시에 위에서 생성한 inference pipeline model을 지정하고 배치 추론을 합니다. confusion matrix 및 f1 score 등을 통해서 모델을 평가 합니다.


---
- 이 워크샵은 아래 블로그들을 참조하여 작성을 함.
## Reference:
    * Blog: Visualizing Amazon SageMaker machine learning predictions with Amazon QuickSight
        * https://aws.amazon.com/blogs/machine-learning/making-machine-learning-predictions-in-amazon-quicksight-and-amazon-sagemaker/
        * Git
            * https://github.com/aws-samples/quicksight-sagemaker-integration-blog
            
    * Blog: Preprocess input data before making predictions using Amazon SageMaker inference pipelines and Scikit-learn
        * https://aws.amazon.com/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/
        * Git: Inference Pipeline with Scikit-learn and Linear Learner
            * https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb