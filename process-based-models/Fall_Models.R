
  # test run for WM
  #par = as.numeric(c(20,5,12))
  #' White's Model as defined in 
  #' White, Thornton, and Running 1997 (Global. Biogeochem. Cy.)
  #' \dontrun{
  #' estimate = WM(data = data, par = par)
  #' }
  WM <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    Tb1 <- par[1]
    Tb2 <- par[2]
    Pb <- par[3]
    # start at doy 182 (July 1st)
    t0 <- which(data$doy == 182)
    
    # photoperiod trigger
    Pt <- ifelse(data$Li < Pb, data$Li - Pb, 0)
    Pt[1:t0,] <- 0
    
    # Tbase 1 trigger
    Tb1t <- ifelse(data$Ti < Tb1, data$Ti - Tb1, 0)
    Tb1t[1:t0,] <- 0
    Pt[Tb1t >= 0] <- 0
    
    # Tbase 2 trigger
    Tb2t <- ifelse(data$Ti < Tb2, data$Ti - Tb2, 0)
    Tb2t[1:t0,] <- 0
    #
    
    # DOY of senescence criterium      
    doy_Pt <- apply(Pt,2, function(xt){
      data$doy[which(xt != 0)[1]]
    })
    doy_Pt[is.na(doy_Pt)] <- 9999
    doy_Tb2t <- apply(Tb2t,2, function(xt){
      data$doy[which(xt != 0)[1]]
    })
    doy_Tb2t[is.na(doy_Tb2t)] <- 9999
    # get first predicted senescence based on either trigger 
    doy = apply(rbind(doy_Pt, doy_Tb2t),2, function(x){
      x[which.min(x)]
    })
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  
  
  # test run for DM
  #par = as.numeric(c(30,1,1,1000,13))
  #' Delpierre's Model as defined in 
  #' Delpierre, Dufrêne, Soudani, Ulrich, Cecchini, Boé, and François 2009  (Agr. For. Met.)
  #' with a monotonously decreasing function weakened with shorter days
  #' \dontrun{
  #' estimate = DM(data = data, par = par)
  #' }
  DM1 <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    Tb <- par[1]
    x <- round(par[2]) # can take discrete values {0,1,2}
    y <- round(par[3]) # can take discrete values {0,1,2}
    F_crit <- par[4]
    P_start <- par[5]
    
    # t0 effectively varies with latitude according to P_start,
    # though set values prior to doy 182 (July 1st) to 0
    # for computational speed
    t0 <- which(data$doy == 182)
    
    # accumulate chilling effects once photoperiod is sufficiently short, using first 
    # photoperiod function option
    Rs <- ifelse(data$Li < P_start & data$Ti < Tb, ((Tb - data$Ti)^x)*((data$Li/P_start)^y), 0)
    Rs[1:t0,] <- 0
    
    # DOY of senescence criterium      
    doy <- apply(Rs,2, function(xt){
      data$doy[which(cumsum(xt) >= F_crit)[1]]
    })
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  
  # test run for DM
  #par = as.numeric(c(30,1,1,1000,13))
  #' Delpierre's Model as defined in 
  #' Delpierre, Dufrêne, Soudani, Ulrich, Cecchini, Boé, and François 2009  (Agr. For. Met.)
  #' with a monotonously decreasing function amplified with shorter days
  #' \dontrun{
  #' estimate = DM(data = data, par = par)
  #' }
  DM2 <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    Tb <- par[1]
    x <- round(par[2]) # can take discrete values {0,1,2}
    y <- round(par[3]) # can take discrete values {0,1,2}
    F_crit <- par[4]
    P_start <- par[5]
    
    # t0 effectively varies with latitude according to P_start,
    # though set values prior to doy 182 (July 1st) to 0
    # for computational speed
    t0 <- which(data$doy == 182)
    
    # accumulate chilling effects once photoperiod is sufficiently short, using second 
    # photoperiod function option
    Rs <- ifelse(data$Li < P_start & data$Ti < Tb, ((Tb - data$Ti)^x)*((1 - (data$Li/P_start))^y), 0)
    Rs[1:t0,] <- 0
    
    # DOY of senescence criterium      
    doy <- apply(Rs,2, function(xt){
      data$doy[which(cumsum(xt) >= F_crit)[1]]
    })
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  
  
  # test run for DM
  #par = as.numeric(c(30,1,1,1000,13))
  #' Delpierre's Model as defined in 
  #' Delpierre, Dufrêne, Soudani, Ulrich, Cecchini, Boé, and François 2009  (Agr. For. Met.)
  #' with x = 1 and y = 1 following Zani et al. (2020)
  #' \dontrun{
  #' estimate = DM(data = data, par = par)
  #' }
  DM1Za20 <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    Tb <- par[1]
    x <- 1
    y <- 1
    F_crit <- par[2]
    P_start <- par[3]
    
    # t0 effectively varies with latitude according to P_start,
    # though set values prior to doy 182 (July 1st) to 0
    # for computational speed
    t0 <- which(data$doy == 182)
    
    # accumulate chilling effects once photoperiod is sufficiently short, using first 
    # photoperiod function option
    Rs <- ifelse(data$Li < P_start & data$Ti < Tb, ((Tb - data$Ti)^x)*((data$Li/P_start)^y), 0)
    Rs[1:t0,] <- 0
    
    # DOY of senescence criterium      
    doy <- apply(Rs,2, function(xt){
      data$doy[which(cumsum(xt) >= F_crit)[1]]
    })
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  
  
  # test run for JM
  # par = as.numeric(c(20,100,13))
  #' Jeong's Model as defined in 
  #' Jeong and Medvigy 2014  (Glob. Eco. & Biogeog.)
  #' \dontrun{
  #' estimate = JM(data = data, par = par)
  #' }
  JM <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    Tb <- par[1]
    F_crit <- par[2]
    P_start <- par[3]
    
    # t0 effectively varies with latitude according to P_start,
    # though set values prior to doy 182 (July 1st) to 0
    # for computational speed
    t0 <- which(data$doy == 182)
    
    # accumulate chilling effects once photoperiod is sufficiently short
    CDD_i <- ifelse(data$Li < P_start & data$Ti < Tb,Tb - data$Ti, 0)
    CDD_i[1:t0,] <- 0
    
    # DOY of senescence criterium      
    doy <- apply(CDD_i,2, function(xt){
      data$doy[which(cumsum(xt) >= F_crit)[1]]
    })
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  
  
  
  # test run for DPDI
  # par = as.numeric(c(0,50,13, 50))
  #' DormPhot Dormancy Induction model as defined in
  #' Caffarra, Donnelly and Chuine 2011 (Clim. Res.)
  #' parameter ranges are taken from Basler et al. 2016
  #' \dontrun{
  #' estimate = DPDI(data = data, par = par)
  #'}
  DPDI <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    a <- par[1]
    b <- par[2]
    L_crit <- par[3]
    D_crit <- par[4]
    
    # set the t0 value if necessary (Jul. 1)
    t0 <- which(data$doy == 182)
    
    # dormancy induction
    # (this is all vectorized doing cell by cell multiplications with
    # the sub matrices in the nested list)
    DR <- 1/(1 + exp(a * (data$Ti - b))) * 1/(1 + exp(10 * (data$Li - L_crit)))
    if (!length(t0) == 0){
      DR[1:t0,] <- 0
    }
    DS <- apply(DR,2, cumsum)
    
    # DOY of senescence criterium      
    doy <- apply(DS,2, function(xt){
      data$doy[which(cumsum(xt) >= D_crit)[1]]
    })
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
    
  }
  
  
  
  # test run for DMs
  # par = as.numeric(c(30,1,1,1000,2,13))
  #' Delpierre's Model with carryover effect of timing of preceding leaf emergence as defined in 
  #' Delpierre, Dufrêne, Soudani, Ulrich, Cecchini, Boé, and François 2009  (Agr. For. Met.)
  #' and
  #' Liu, Piao, Campioli, Gao, Fu, Wang, He, Li, and Janssens 2020 (Glob. Chg. Bio.)
  #' \dontrun{
  #' estimate = DM(data = data, par = par)
  #' }
  DM1s <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    Tb <- par[1]
    x <- round(par[2]) # can take discrete values {0,1,2}
    y <- round(par[3]) # can take discrete values {0,1,2}
    a_Sa <- par[4]
    b_Sa <- par[5]
    P_start <- par[6]
    
    # t0 effectively varies with latitude according to P_start,
    # though set values prior to doy 182 (July 1st) to 0
    # for computational speed
    t0 <- which(data$doy == 182)
    
    # accumulate chilling effects once photoperiod is sufficiently short, using first 
    # photoperiod function option
    Rs <- ifelse(data$Li < P_start & data$Ti < Tb, ((Tb - data$Ti)^x)*((data$Li/P_start)^y), 0)
    Rs[1:t0,] <- 0
    
    # determine Sa, the anomaly of the preceding spring leaf unfolding date 
    # relative to the long-term mean
    # Use TT to predict SOS and save this in each phenor driving list
    
    Sa = data$SOS_TT_Sa_vs_LTM
    if(length(which(is.na(Sa)==T))>=1){
      Sa[which(is.na(Sa))] = 0
    }
    
    
    # determine F_crit
    F_crit = a_Sa + b_Sa*Sa
    
    # DOY of senescence criterium      
    doy <- c()
    for(i in 1:length(F_crit)){
      doy <- c(doy, data$doy[which(cumsum(Rs[,i]) >= F_crit[i])[1]])
    }
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  
  
  # test run for DPDIs
  # par = as.numeric(c(0,50,13, 50))
  #' DormPhot Dormancy Induction model with carryover effect of timing of preceding leaf emergence as defined in
  #' Caffarra, Donnelly and Chuine 2011 (Clim. Res.)
  #' and
  #' Liu, Piao, Campioli, Gao, Fu, Wang, He, Li, and Janssens 2020 (Glob. Chg. Bio.)
  #' parameter ranges are taken from Basler et al. 2016
  #' \dontrun{
  #' estimate = DPDIs(data = data, par = par)
  #'}
  DPDIs <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    a <- par[1]
    b <- par[2]
    a_Sa <- par[3]
    b_Sa <- par[4]
    L_crit <- par[5]
    
    # set the t0 value if necessary (Jul. 1)
    t0 <- which(data$doy == 182)
    
    # dormancy induction
    # (this is all vectorized doing cell by cell multiplications with
    # the sub matrices in the nested list)
    DR <- 1/(1 + exp(a * (data$Ti - b))) * 1/(1 + exp(10 * (data$Li - L_crit)))
    if (!length(t0) == 0){
      DR[1:t0,] <- 0
    }
    DS <- apply(DR,2, cumsum)
    
    # determine Sa, the anomaly of the preceding spring leaf unfolding date 
    # relative to the long-term mean
    # Use TT to predict SOS and save this in each phenor driving list
    
    Sa = data$SOS_TT_Sa_vs_LTM
    if(length(which(is.na(Sa)==T))>=1){
      Sa[which(is.na(Sa))] = 0
    }
    
    
    # determine D_crit
    D_crit = a_Sa + b_Sa*Sa
    
    
    # DOY of senescence criterium      
    doy <- c()
    for(i in 1:length(D_crit)){
      doy <- c(doy, data$doy[which(cumsum(DS[,i]) >= D_crit[i])[1]])
    }
    
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
    
  }
  
  # test run for PIAG
  # par = as.numeric(c(0.05, 125, 150, 2.5, 12))
  #' Photosynthesis Influenced Autumn Model as defined in 
  #' Zani et al 2020  (Sci.)
  #' \dontrun{
  #' estimate = PIAG(data = data, par = par)
  #' }
  PIAG <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    a <- par[1] # for RLSTPM accumulation, ranges 0 - 0.1 in Meier and Bigler, 2023
    b <- par[2] # for RLSTPM accumulation, ranges 0 - 250 in Meier and Bigler, 2023
    n <- par[3] # for Fcrit calculation, ranges 0 - 300 in Meier and Bigler, 2023 (therein "b0")
    o <- par[4] # for Fcrit calculation, ranges 0 - 5 in Meier and Bigler, 2023 (therein "b0")
    P_start <- par[5] # ranges 10 - 20 in Meier and Bigler, 2023, I chose 8 - 16 to reflect range in obs. data

    # photoperiod-dependent start-date for chilling accumulation (t0)
    # t0 is defined as the first day when photoperiod is shorter than the photoperiod threshold (P_start)
    # after the date of the longest photoperiod (summer solstice),the 173rd day of year
    t0 <- which(data$doy == 173)
    t0_P_start = apply(data$Li[(t0+1):365,], 2, function(xt){
      which(xt <= P_start)[1]
    })
    if(length(which(is.na(t0_P_start)))>=1){
      t0_P_start[which(is.na(t0_P_start))] = 365-t0
    }
    
    t0_P_start = t0 + t0_P_start
    
    # accumulate cooling and shortening daylength effects
    Rs = 1/(1+exp(a*(data$Tmini*data$Li-b)))
    for(t0_i in 1:length(t0_P_start)){
      Rs[1:t0_P_start[t0_i],t0_i] <- 0
    }
   
    
    # calculate F_crit according to the Growing Season Index (GSI)
    # get GSI from data list
    # get cGSI from accumulation based on estimated time of leaf out
    # note GSI before estimated leaf out set to 0
    cGSI = matrix(data = NA, nrow = 1, ncol = length(data$site))
    for(GSI_i in 1:length(data$site)){
      P_end = which(data$Li[170:365,GSI_i] <= 11)[1] + 169
      cGSI[GSI_i] = sum(data$GSI[1:P_end,GSI_i])
      
    }
    
    F_crit = n + o * cGSI
    
    
    # DOY of senescence criterium 
    doy <- c()
    for(F_i in 1:length(F_crit)){
      
      doy <- c(doy, data$doy[which(cumsum(Rs[,F_i]) >= F_crit[F_i])[1]])
    }

    
    
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  

  
  # test run for DMP (aka TPDMZa20 in Meier and Bigler, 2023)
  # par = as.numeric(c(20,100,13))
  #' Delpierre Model 1 with Precipitation as defined in 
  #' Zani et al 2020  (Sci.) and Meier and Bigler 2023
  #' \dontrun{
  #' estimate = DMP(data = data, par = par)
  #' }
  DMP <- function(par, data){
    # extract the parameter values from the
    # par argument for readability
    a <- par[1] # for RLSTPM accumulation, ranges 0 - 0.1 in Meier and Bigler, 2023
    b <- par[2] # for RLSTPM accumulation, ranges 0 - 250 in Meier and Bigler, 2023
    g <- par[3] # for Fcrit calculation, ranges 0 - 150 in Meier and Bigler, 2023 (therein "b0")
    h <- par[4] # for Fcrit calculation, ranges -15 - +40 in Meier and Bigler, 2023 (therein "b1")
    ii <- par[5] # for Fcrit calculation, ranges -40 - +15 in Meier and Bigler, 2023 (therein "b2", "i" in Zani et al., 2020)
    P_start <- par[6] # ranges 10 - 20 in Meier and Bigler, 2023, I chose 8 - 16 to reflect range in obs. data
    
    
    # photoperiod-dependent start-date for chilling accumulation (t0)
    # t0 is defined as the first day when photoperiod is shorter than the photoperiod threshold (P_start)
    # after the date of the longest photoperiod (summer solstice),the 173rd day of year
    t0 <- which(data$doy == 173)
    t0_P_start = apply(data$Li[(t0+1):365,], 2, function(xt){
      which(xt <= P_start)[1]
    })
    
    if(length(which(is.na(t0_P_start)))>=1){
      t0_P_start[which(is.na(t0_P_start))] = 365-t0
    }
    
    t0_P_start = t0 + t0_P_start
    
    # accumulate cooling and shortening daylength effects
    Rs = 1/(1+exp(a*(data$Tmini*data$Li-b)))
    
    # set Rs value before t0_P_start to 0
    for(t0_i in 1:length(t0_P_start)){
      Rs[1:t0_P_start[t0_i],t0_i] <- 0
    }
    
    
    # calculate F_crit according to the mean summer temp and low precipitation indices
    # get estimated SOS from data list
    # get average minimum summer temp from accumulation based on estimated time of leaf out
    # note temp before estimated leaf out excluded

    Tsummer = matrix(data = NA, nrow = 1, ncol = length(data$site))
    for(Tsummer_i in 1:length(data$site)){
      # This should actually be the site-average LC date according to Meier 2023, but to 
      # avoid circular calculation I am using the period from SOS until Li < 11 hours
      SOS = data$SOS_TT[Tsummer_i]
      P_end = which(data$Li[170:365,Tsummer_i] <= 11)[1] + 169
      Tsummer[Tsummer_i] = mean(data$Tmini[SOS:P_end,Tsummer_i])
  
      
    }
    
    # RDsummer = LPI
    RDsummer = matrix(data = NA, nrow = 1, ncol = length(data$site))
    for(RDsummer_i in 1:length(data$site)){
      # This should actually be the site-average LC date according to Meier 2023, but to 
      # avoid circular calculation I am using the period from SOS until Li < 11
      SOS = data$SOS_TT[RDsummer_i]
      P_end = which(data$Li[170:365,RDsummer_i] <= 11)[1] + 169
      RDsummer[RDsummer_i] = length(which(data$Pi[SOS:P_end,RDsummer_i] <= 2))
      
    }
    
    F_crit = g + h * Tsummer + ii * RDsummer
    
    
    
    # DOY of senescence criterium 
    # if doy = 1 predicted due to negative F_crit, set to 
    # first day of t0 below P_start (more reasonable based on allowable
    # accumulation period)
    doy <- c()
    for(F_i in 1:length(F_crit)){
      d_i = data$doy[which(cumsum(Rs[,F_i]) >= F_crit[F_i])[1]]
      if(is.na(d_i)==T){
        d_i = 9999
      }
      if(d_i >= t0_P_start[F_i]){
        doy <- c(doy, d_i)
      } else {
        doy <- c(doy, t0_P_start[F_i])
      }
      
    }
    
    # set export format, either a rasterLayer
    # or a vector
    shape_model_output(data = data, doy = doy)
  }
  