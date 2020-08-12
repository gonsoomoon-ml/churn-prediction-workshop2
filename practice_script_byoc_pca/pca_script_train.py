
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import logging

from sklearn.decomposition import PCA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--n_components', type=int, default = 3)
    
    args = parser.parse_args()
    
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)
    
    pca = PCA(n_components = args.n_components)
    print("train shape: ", train_data.shape)
    X_new = pca.fit_transform(train_data)
    
    print("Component Variability: \n", pca.explained_variance_ratio_)
    
    joblib.dump(pca, os.path.join(args.model_dir, "model.joblib"))
    

def model_fn(model_dir):
    """
    Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """   
    pca = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return pca  

def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    """
    logging.info(f"predict_fn: input_data - '{input_data}'")
    # model, PCA model, has transform()
    components = model.transform(input_data)
    
    logging.info(f"predict_fn: PCA components: \n'{components}'")    
    return components
    
# predict_fn을 정의하지 않으면 default predict_fn을 호출 함.
# PCA는 predict 함수를 제공하지 않으므로 사용자 정의 필요 함.

# algo-1-dhteh_1  | 2020-08-10 14:15:55,970 ERROR - pca_train - Exception on /invocations [POST]
# algo-1-dhteh_1  | Traceback (most recent call last):
# algo-1-dhteh_1  |   File "/miniconda3/lib/python3.7/site-packages/sagemaker_containers/_functions.py", line 93, in wrapper
# algo-1-dhteh_1  |     return fn(*args, **kwargs)
# algo-1-dhteh_1  |   File "/miniconda3/lib/python3.7/site-packages/sagemaker_sklearn_container/serving.py", line 70, in default_predict_fn
# algo-1-dhteh_1  |     output = model.predict(input_data)
# algo-1-dhteh_1  | AttributeError: 'PCA' object has no attribute 'predict'
