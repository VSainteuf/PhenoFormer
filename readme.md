
# Deep learning meets tree phenology modeling: PhenoFormer vs. process-based models

This repository contains the python code to reproduce the results of the deep learning and traditional machine learning models 
of the paper. 


## Setting up 

### Python environment 

We recommend to create a new virtual environment with `python==3.10` :
```
conda create --name phenoformer python==3.10
conda activate phenofomer
```

To install requirements, use the requirements file:
```setup
pip install -r requirements.txt
```
> ⚠️ If you run into issues , make sure that your pip version is < 24.1 by running:

```setup
pip install pip==24.0
```

### Hardware

For the deep learning scripts, we recommend using a machine with GPU to have reasonable training times. 
Our models are still quite small (for deep learning standards) so a small GPU of even 8GB VRAM would do. 


## PhenoFormer 

There are four scripts to run the different configurations of PhenoFormer:
- `xp-spring-singletask.py` to train the single-species PhenoFormer for spring phenology (variant (a) in Table 6)
- `xp-spring-multitask.py` to train the multi-species variants of PhenoFormer for spring phenology (variants (b->e) in Table 6)
- `xp-autumn-singletask.py` to train the single-species variants of PhenoFormer for autumn phenology (variants (a,f) in Table 9)
- `xp-autumn-multitask.py` to train the multi-species variants of PhenoFormer for autumn phenology (variants (b,d) in Table 9)

### Training 

To run each of these script:
1. Complete the `data_folder` field with the path to the dataset folder on your machine. 
2. Complete the `save_dir` field with the path to the folder where to write the results. 
3. (Optional) Comment/Uncomment the configuration of the variant you would like to train.
4. Activate the proper python environement and run the script.

### Results 

The results of each configuration (model x dataset split) gets written in a separate subdirectory in `save_dir`.
The scripts will execute the 10 runs for each configuration by default. 
So in each subdirectory you will find one result file per fold with the following name `run_summary_foldX.json` .
The result files contain all the values of the different hyperparameters as well as the performance metrics on the train, validation, and test sets. 

## Traditional Machine Learning baselines 

The present repository also contains the code to reproduce the results of the traditional machine learning baselines. 

To do so use the `xp-baselines-traditional-ml.py` script and follow these steps:
1. Complete the `data_folder` field with the path to the dataset folder on your machine. 
2. Complete the `save_dir` field with the path to the folder where to write the results. 
3. Comment/Uncomment the configuration of the variant you would like to train.
4. Activate the proper python environement and run the script.


