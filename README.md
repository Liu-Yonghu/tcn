data_preprocess:

    this module using to clean the data

tcn_model.py:

    definition of:
        - TCN model simple (only dilation block)
        - TCN model complex (dialtion block + normalization layers + residual block)
        - Capsule model
        - LTC model
        - NCP model

NAS_custom.py:

    Customized autokeras NAS controller (either GridSearch or
    BayesianOptimization).

    By default, the controller searches for the model with the 
    **minimum mean validation error** among multiple executions.

    The custom one instead uses the training and validation splits 
    to train the models, the test split to evaluate the model, 
    and returns the best model with the **min test error** among 
    multiple trials and multiple executions.

utils.py:

    Utility functions such as:
        - how to create and split dataset
        - how to compute and save test
        - how to evaluate SPARC
        - other model infos
# runing demo 
'''
python main.py --data_path 'exp_data/TOF1.csv' --output_dir 'results/segment/1'
'''
