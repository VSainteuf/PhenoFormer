
<div align="center">
<h1>PhenoFormer: deep learning for species-level tree phenology</h1>
</div>



This repository contains the code to reproduce the results presented in the article *Deep learning meets tree phenology modeling: PhenoFormer vs. process-based models*, Garnot et al. published in Methods in Ecology and Evolution. 

The present repository contains :
- the PyTorch implementation of PhenoFormer (`model/architecture`)
- the python scripts used to train the different variants of PhenoFormer presented in the paper (`run-phenoformer-x-x.py`).
- the R scripts used to evaluate the performance of process-based models on our dataset (see `/process-based-models`).
- the python scipts used to train the traditional machine learning baselines used in the paper (`run-traditional-ml-baselines.py`)
- our detailed numerical results (`results/full_results.csv`) 

## 0. Setting up 

### Dowload the dataset 
You can retrieve our dataset from our [Zenodo archive](https://zenodo.org/records/15045780). This dataset contains two subfolders: one version of the dataset formatted for R scripts and the other one for python scripts. Please use the `learning-models-data` subfolder for all python scripts. 

### Clone the repository and install requirements
```
git clone git@github.com:VSainteuf/PhenoFormer.git
cd PhenoFormer
conda create --name phenoformer python==3.10
conda activate phenoformer
pip install -r requirements.txt
```
We recommend to create a new virtual environment with `python==3.10` :

> ⚠️ If you run into issues , make sure that your pip version is < 24.1 by running:

```setup
pip install pip==24.0
```

### Hardware

For the deep learning scripts, we recommend using a machine with GPU to have reasonable training times. Our models are still quite small (for deep learning standards) so a small GPU of even 4 or 8GB VRAM would do. 


## 1. PhenoFormer 

There are four scripts to run the different configurations of PhenoFormer:
- `run-phenoformer-singlespecies-spring.py` to train the single-species PhenoFormer for spring phenology (variant (a) in Table 6)
- `run-phenoformer-multispecies-spring.py` to train the multi-species variants of PhenoFormer for spring phenology (variants (b->e) in Table 6)
- `run-phenoformer-singlespecies-autumn.py` to train the single-species variants of PhenoFormer for autumn phenology (variants (a,f) in Table 9)
- `run-phenoformer-multispecies-autumn.py` to train the multi-species variants of PhenoFormer for autumn phenology (variants (b,d) in Table 9)

#### Training 

To run each of these script:
1. Complete the `data_folder` field with the path to the `learning-models-data` dataset folder on your machine. 
2. Complete the `save_dir` field with the path to the folder where to write the results. 
3. (Optional) Comment/Uncomment the configuration of the variant you would like to train.
4. Activate the proper python environement and run the script.

#### Results 

The results of each configuration (model x dataset split) gets written in a separate subdirectory in `save_dir`.
The scripts will execute the 40 runs for each configuration by default. 
So in each subdirectory you will find one result file per fold with the following name `run_summary_foldX.json` .
The result files contain all the values of the different hyperparameters as well as the performance metrics on the train, validation, and test sets. 

## 2. Traditional machine learning baselines 

The present repository also contains the code to reproduce the results of the traditional machine learning baselines. 

To do so use the `run-traditional-ml-baselines.py` script and follow these steps:
1. Complete the `data_folder` field with the path to the `learning-models-data` dataset folder on your machine. 
2. Complete the `save_dir` field with the path to the folder where to write the results. 
3. Comment/Uncomment the configuration of the variant you would like to train.
4. Activate the proper python environement and run the script.


## Credits 

To cite this work please use:
```bibtex
@article{phenoformer2025,  
  title={Deep learning meets tree phenology modeling: PhenoFormer vs. process-based models},  
  author={Garnot, Vivien Sainte Fare and Spafford, Lynsay and Lever, Jelle and 
  Sigg, Christian and Pietragalla, Barbara and Vitasse, Yann 
  and Gessler, Arthur and Wegner, Jan Dirk},  
  journal={{Methods in Ecology and Evolution}},  
  year={2025}  
}  
```

- Data source: Federal Office of Meteorology and Climatology (MeteoSwiss)  
- Meteorological data processing: Swiss Federal Institute for Forest, 
Snow and Landscape Research (WSL)
- This project was co-financed by the Federal Office for Meteorology and Climatology MeteoSwiss within the framework of GAW-CH.
