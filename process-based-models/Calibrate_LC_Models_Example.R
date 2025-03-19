#-------------------------------------------------
# Preamble: This script calibrates plant phenology
# models in phenor using ground observations 
# from MeteoSwiss with a selection of training
# and testing site-years based on a json file 
# specifying train-test splits.

#-------------------------------------------------
# I. Clear environment.
#-------------------------------------------------
gc()
rm(list=ls(all=TRUE))
#-------------------------------------------------
# II. Install and load libraries as necessary.
#-------------------------------------------------
if(!require('phenor')) install.packages('phenor'); library(phenor)
if(!require('stringr')) install.packages('stringr'); library(stringr)
if(!require('plyr')) install.packages('plyr'); library(plyr)
if(!require('dplyr')) install.packages('dplyr'); library(dplyr)
if(!require('RJSONIO')) install.packages('RJSONIO'); library(RJSONIO)

#-------------------------------------------------
# III. Define constants and functions.
#-------------------------------------------------
# Define path to folder where:
# 1. training data are stored in the form of ".RDS" files,
# 2. parameter range files are stored in the form of .csv files, 
# 3. train-test splits are stored in the form of JSON files. 
path_to_data = "path/to/data/"
path_to_split_files = "path/to/phenoformer/splits"

# Define path to folder where coding scripts are stored, in 
# order to load R model functions for autumn/fall models.
path_to_code = "path/to/code"
source(paste0(path_to_code,"/Fall_Models.R"))

# Define species of interest
sp = "European_beech"

# Here specify the training and testing split of interest, 
# either "uniformly-rdm", "site-random", "year-random",
# or "structured-temporal"
split = "uniformly-rdm"
if(split == "uniformly-rdm"){
training_config = "uniformly_random_split"
json_file = paste0(path_to_split_files,"/uniformly-rdm-split.json")
json_data = fromJSON(json_file)
}
if(split == "site-random"){
  training_config = "random_spatial_split"
  json_file = paste0(path_to_split_files,"/rdm-spatial-split.json")
  json_data = fromJSON(json_file)
}
if(split == "year-random"){
  training_config = "random_temporal_split"
  json_file = paste0(path_to_split_files,"/rdm-temporal-split.json")
  json_data = fromJSON(json_file)
}
if(split == "structured-temporal"){
  training_config = "structured_temporal_split"
  
}
# Define path to phenor parameter ranges, with Tbase for forcing > 0°C, and t0 for 
# forcing accumulation greater than Jan. 1
par_ranges = paste0(path_to_data,"/parameter_ranges_fall.csv")

# Define leaf colouration models of interest.
fall_models = c("WM","DM1","DM2","DM1Za20","JM","DPDI","DM1s","DPDIs","PIAG","DMP", "NULL")

# Use Hufkens et al. (2018) parameterization approach:
control = list(temperature = 10000, max.call = 40000)

#-------------------------------------------------
# IV. Load species training data.
#-------------------------------------------------
# Collect names of training lists.
training_lists <- list.files(path_to_data, pattern = ".RDS")
# Collect species order of training lists.
species <- unlist(lapply(training_lists, function(x) str_split(str_split(x,"list_")[[1]][2],".RDS")[[1]][1]))
# Collect phenophase order of training lists.
phenophases <- unlist(lapply(training_lists, function(x) str_split(str_split(x,"ground_")[[1]][2],"_nested")[[1]][1]))
# Filter to phenophase and species of interest.
index = which(phenophases %in% c("leaf_colouring","needle_colouring")&(species == sp))
# Load phenor list with all phenology observations and daily data.
pooled_list = readRDS(paste0(path_to_data,"/",training_lists[index]))


#-------------------------------------------------
# V. Calibrate and test leaf colouration phenor models.
#-------------------------------------------------
# establish metrics of interest to record with each calibration and test.
optimal_params = list()
model_col = c()
species_col = c()
RMSE_col = c()
fold_col = c()
R2_col = c()

# Designate training and testing site-years based on train-test splits
for(k in 1:40){
  
  # training site-years
  if(split != "structured-temporal"){
    training_site_years = json_data[[k]]$train
    training_index = c()
    for(ti in 1:length(training_site_years)){
      site = str_sub(training_site_years[ti],end=-6)
      year = str_sub(training_site_years[ti],-4,-1)
      ind = which((pooled_list$site == site)&(pooled_list$year == year))
      if(length(ind) == 1){
        training_index = c(training_index, ind)
      } 
      
    }
    
    # testing site-years
    testing_site_years = json_data[[k]]$test
    testing_index = c()
    for(vi in 1:length(testing_site_years)){
      site = str_sub(testing_site_years[vi],end=-6)
      year = str_sub(testing_site_years[vi],-4,-1)
      ind = which((pooled_list$site == site)&(pooled_list$year == year))
      if(length(ind) == 1){
        testing_index = c(testing_index, ind)
      } 
      
    }
    
  } else {
    training_index = which(pooled_list$year <= 2002)
    testing_index = which(pooled_list$year >= 2013)
  }
  
  # Establish training list
  training = pooled_list
  list_items = names(training)
  for(i in 1:length(list_items)){
    item = training[[i]]
    name = list_items[i]
    if(name == "doy"){
      next()
    }
    if(length(item)%in% c(0,1)){
      next()
    }
    
    dims = dim(item)
    if(is.null(dims)){
      item = item[training_index]
      training[[i]] = item
      next()
    }
    if(dims[1]==1){
      item = item[training_index]
    } else {
      item = item[,training_index]
    }
    training[[i]] = item
  }
  
  # Establish testing list
  testing = pooled_list
  list_items = names(testing)
  for(i in 1:length(list_items)){
    item = testing[[i]]
    name = list_items[i]
    if(name == "doy"){
      next()
    }
    if(length(item)%in% c(0,1)){
      next()
    }
    
    dims = dim(item)
    if(is.null(dims)){
      item = item[testing_index]
      testing[[i]] = item
      next()
    }
    if(dims[1]==1){
      item = item[testing_index]
    } else {
      item = item[,testing_index]
    }
    testing[[i]] = item
  }
  
  
  # Train and Test
  for(model in fall_models){
    if(model != "NULL"){
      # Train
      d <- c()
      d <- pr_parameters(model = model, par_ranges = par_ranges)
      upper = as.numeric(d[2,])
      lower = as.numeric(d[1,])
      optim.par <- pr_fit_parameters(par = NULL, data = training, cost = rmse, model = model, method = "GenSA", lower = lower, upper = upper, control = control)
      optimal_params[[model]][[sp]][[k]] <- optim.par$par
      
      # Test
      out <- pr_predict(optim.par$par, data = testing, model = model)
      if(max(as.numeric(out),na.rm=T) >= 365){ # this may occur with a small max.call in control
        index = which(as.numeric(out) >= 365)
        out[index] = NA
      }
      
      # Account for non-predictions
      NA_out = which(is.na(out)==T) # this may occur with a small max.call in control
      
      if(length(NA_out)>=1){
        RMSE <- sqrt(mean((out[-NA_out] - testing$transition_dates[-NA_out])^2,na.rm=T))
        rss <- sum((out[-NA_out] - testing$transition_dates[-NA_out]) ^ 2)  ## residual sum of squares
        tss <- sum((testing$transition_dates[-NA_out] - mean(testing$transition_dates[-NA_out])) ^ 2)  ## total sum of squares
        rsq <- 1 - rss/tss
        
      } else {
        RMSE <- sqrt(mean((out - testing$transition_dates)^2,na.rm=T))
        rss <- sum((out - testing$transition_dates) ^ 2)  ## residual sum of squares
        tss <- sum((testing$transition_dates - mean(testing$transition_dates, na.rm = T)) ^ 2)  ## total sum of squares
        rsq <- 1 - rss/tss
        
      }
      
    } else {
      
      out = rep(mean(training$transition_dates, na.rm = T),length(testing$site))
      
      RMSE <- sqrt(mean((out - testing$transition_dates)^2,na.rm=T))
      rss <- sum((out - testing$transition_dates) ^ 2)  ## residual sum of squares
      tss <- sum((testing$transition_dates - mean(testing$transition_dates)) ^ 2)  ## total sum of squares
      rsq <- 1 - rss/tss
      
    }
    
    # Print calibration progress
    print("")
    print(Sys.time())
    print(training_config)
    print(sp)
    print(model)
    print(paste("Fold:",k))
    print(paste("RMSE:",RMSE))
    print(paste("Coefficient of Determination (R^2):",rsq))
    
    
    # Store metrics
    model_col = c(model_col, model)
    species_col = c(species_col, sp)
    RMSE_col = c(RMSE_col, RMSE)
    fold_col = c(fold_col, k)
    R2_col = c(R2_col, rsq)
  }
  
}

#-------------------------------------------------
# VI. Store output. 
#-------------------------------------------------
path_to_output = "path/to/output/"
# RMSE
RMSE_df = cbind.data.frame(model_col, species_col, RMSE_col,R2_col, fold_col)
write.csv(RMSE_df, file = paste0(path_to_output,"/",training_config,"_",sp,"_RMSE_LC_models.csv"), row.names = F)
# optimal parameters
saveRDS(optimal_params, file = paste0(path_to_output,"/",training_config,"_",sp,"_optimal_params_LC_models.RDS"))


