
FROM 366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3
    
# install python package
RUN pip install joblib


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/ml/code:${PATH}"

# Copy training code
COPY code/* /opt/ml/code/
 
WORKDIR /opt/ml/code

# ENTRYPOINT ["python", "pca_train.py"]
# In order to use SageMaker Env varaibles, use the statement below
ENV SAGEMAKER_PROGRAM pca_byoc_train.py
