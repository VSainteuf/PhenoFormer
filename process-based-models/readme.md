# Process-based models scripts

R codes used to train the process-based models used in our study for leaf emergence (LE) and lead colouration (LC). 


### Running the R code

1️⃣ **Download the dataset** from our [Zenodo archive](https://zenodo.org/records/15045780)

2️⃣ **Specify the paths** on your local machine:
-  to the dataset by updating the variable `path_to_data` line 29 of the R script. (please specify the path to the `process-models-data` subfolder of the Zenodo archive)
-  to the folder of the present repository that contains the json files specifying how the dataset is split for each fold of each configuration by updating variable `path_to_split_files` line 30 of the script (`path_to_split_files=/local/path/to/phenoformer/splits`).
- to the output folder where results will be stored, by updating the variable `path_to_output` at the end of the script. 
- (Only for leaf colouration script) specify the path to the script `Fall_Models.R`  specifying additional colouration models by updating the variable on line 33 of the script. 

3️⃣ **Select a configuration** For both scripts, the user needs to define which dataset split to use on line 37 or 42 for the leaf emergence and leaf colouration script, respectively. The options are as follows: "uniformly-rdm", "site-random", "year-random", or "structured-temporal", which correspond to splits 1, 2, 3, and 4, respectively, in the article.

4️⃣ **Run the script** Once all paths are defined, the user can simply run the entire script, and the script will output a .csv with model performance results to the output directory. 
