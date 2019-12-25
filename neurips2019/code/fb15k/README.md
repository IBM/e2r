This directory contains code and data used for experiment on FB15K dataset.

After downloading (and before running any of the scripts here), please unzip 'data.zip'. This should create 'data/' directory containing data needed for error-free execution of the scripts.

Simply running the script 'reasonE.test.py' should use trained model as stored in directory 'model/' and generate accuracy numbers as reported in the paper.

Script 'reasonE.train.py' is for fresh training. Script 'reasonE.retrain.py' is to retrain the latest trained model starting from the point where last training ended.
