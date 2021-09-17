Patient Outcome Prediction Modeling
===================================

Project Organization
----------------------------
     ├── README.md                        <- The summary file of this project
     ├── data                             <- A temporary location to hold data files, not in CodeCommit
     ├── docs                             <- The Sphinx documentation folder
     ├── ml                               <- Source code location that contains all the machine learning modules
     │   ├── config.py                    <- This file defines contant variables
     │   ├── data.py                      <- This file contains helper functions reading from s3, loading embedding matrix, df->tensors etc
     │   ├── evaluation.py                <- Evaluation functions used by model training 
     │   ├── evaluation_tf.py             <- Evaluation functions specifically for TensorFlow framework
     │   ├── requirements.txt             <- Requirements for SageMaker hosted container
     │   └── visualize.py                 <- Visualization function to analyze the data
     ├── models                           <- A temporary location to store the embedding model and trained ml models, not in CodeCommit
     ├── notebooks                        <- A folder containing notebook experiments
     └── requirements.txt                 <- The project dependencies. Run `pip install -r requirements.txt` before you start