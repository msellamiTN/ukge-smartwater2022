# Embedding Uncertain Knowledge Graphs

This repository includes the code of UKGE and data used in the experiments.

## Install
Make sure your local environment has the following installed:

    Python3
    tensorflow >= 1.5.0
    scikit-learn
    
Install the dependents using:

    pip install -r requirements.txt

## Run the experiments
To run the experiments, use:

    python ./run/run.py

or

    python ./run/run.py --data water --model rect --batch_size 1024 --dim 128 --epoch 100 --reg_scale 5e-4
You can use `--model logi` to switch to the UKGE(logi) model.

  
 
