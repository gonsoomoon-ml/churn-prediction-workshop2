

from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import logging
from io import StringIO

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


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
    
def input_fn(input_data, request_content_type):
    """
    기본적인 입력은 'text/csv' 를 받고, 데이타는 타입이 String 혹은 Bytes 를 받습니다.
    Parse input data payload    
    We currently only take csv input. 
    """
  
    print("### input_fn: Starting ###")
    print("type of input_data: ", type(input_data))
    print("request_content_type: ", request_content_type)
        
    content_type = request_content_type.lower(
    ) if request_content_type else "text/csv"
    content_type = content_type.split(";")[0].strip()
        
    
    if isinstance(input_data, str):
        str_buffer = input_data
    else:
        # In the case of getting byte array like b'hello'        
        str_buffer = str(input_data,'utf-8') 

    
    if (content_type == 'text/csv' or content_type == 'text/csv; charset=utf-8'):
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(str_buffer),  header=None)
        logging.info(f"input_fn: ")      
        logging.info(f"shape of requested data: '{df.shape}'")        
        logging.info(f"requested data: '{df}'")
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))    

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
    import os
    
    print("prdict_fn: Staring")
    print("os.getenv('TRANSFORM_MODE'): ", os.getenv('TRANSFORM_MODE'))    

    
        
    if os.getenv('TRANSFORM_MODE') == 'feature-transform':        

        import numpy as np
        print("predcit_fn - TRANSFORM_MODE: ", os.getenv('TRANSFORM_MODE'))
        logging.info(f"predict_fn: input_data - '{input_data}'")
        # model, PCA model, has transform()
        print("type of input_data: ", type(input_data))
        print("shape of input_data: ", input_data.shape)        
        print("head of input_data: \n ", input_data[0:2])  
        payload = input_data.iloc[:,1:] # Exclude a label

        print("os.getenv('LENGTH_COLS') : ", os.getenv('LENGTH_COLS'))
        print("type: os.getenv('LENGTH_COLS'): ", type(os.getenv('LENGTH_COLS')))
        num_cols = int(os.getenv('LENGTH_COLS')) - 1 # exclude a label

        payload = payload.values.reshape(-1,num_cols)
        components = model.transform(payload)

        print("type of components: ", type(components))
        print("shape of components: ", components.shape)

        # Add label column to the front
        features = np.insert(components, 0, input_data.iloc[:,0].values, axis=1)

        logging.info(f"predict_fn: PCA components: \n'{features}'")    

        return features
    
        
    # In th ecase of not being set to env. variable        
    if os.getenv('TRANSFORM_MODE') == 'inverse-label-transform':
        print("predcit_fn - TRANSFORM_MODE: ", os.getenv('TRANSFORM_MODE'))
        # model, PCA model, has transform()
        print("type of input_data: ", type(input_data))
        print("shape of input_data: ", input_data.shape)        
        print("head of input_data: \n ", input_data[0:2])  
        
        payload = input_data

        # num_cols = int(os.getenv('LENGTH_COLS'))
        num_cols = 69

        payload = payload.values.reshape(-1,num_cols)
        features = model.transform(payload)

        logging.info(f"predict_fn: PCA components: \n'{features}'")    

        return features

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    
    import numpy as np
    
    logging.info(f"Output_fn: prdiction - '{prediction}' ")                
    print("Output_fn-type of accept : ", accept )
    
    accept = 'text/csv'
    if type(prediction) is not np.ndarray:
        prediction=prediction.toarray()
    
    print("output_fn-type of prediction: ", type(prediction))    
   
    if accept == 'text/csv':
        print("type of accept after a change : ", accept )        
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        
    
# predict_fn을 정의하지 않으면 default predict_fn을 호출 함.
# PCA는 predict 함수를 제공하지 않으므로 사용자 정의 필요 함.

# algo-1-dhteh_1  | 2020-08-10 14:15:55,970 ERROR - pca_train - Exception on /invocations [POST]
# algo-1-dhteh_1  | Traceback (most recent call last):
# algo-1-dhteh_1  |   File "/miniconda3/lib/python3.7/site-packages/sagemaker_containers/_functions.py", line 93, in wrapper
# algo-1-dhteh_1  |     return fn(*args, **kwargs)
# algo-1-dhteh_1  |   File "/miniconda3/lib/python3.7/site-packages/sagemaker_sklearn_container/serving.py", line 70, in default_predict_fn
# algo-1-dhteh_1  |     output = model.predict(input_data)
# algo-1-dhteh_1  | AttributeError: 'PCA' object has no attribute 'predict'
