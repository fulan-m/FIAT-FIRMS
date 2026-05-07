<div align="center">
  
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Mona+Sans&weight=900&size=30&letterSpacing=4px&duration=1000&pause=200&color=F7F7F7&center=true&vCenter=true&repeat=false&width=435&lines=C.O.M.B.U.S.T.I.O.N.)](https://git.io/typing-svg)
---
# (C)lassification of (O)utliers and (M)odeling (B)urn-risk (U)sing (S)tochastic (T)ree (I)ncident (O)utput (N)etworks

</div>

<div align="justify">
  
# About
This project was developed with the funding under a Research Initiation (IC) scholarship from the São Paulo Research Foundation (FAPESP) during 2025/2026, and presented at the World Symposium on Artificial Intelligence for Climate Change Adaptation and Mitigation (WS-AI4CCAM) at the _Fluminense_ Federal University (UFF), Niterói - Rio de Janeiro, Brazil.
Using Python, the script uses the Random Forest Classifier from [Scikit-Learn's](https://scikit-learn.org/) package to train a fire prediction model to assess the probability of Fire Incident / Anomalous Temperature events (FIATs) inside the Cerrado biome of São Paulo.

The datasets used for all analyis conducted on the paper are not available in this repository, but the script to generate them using the Google Earth Engine API is available [here](https://github.com/fulan-m/AnomaFire/blob/main/export_created_datasets.ipynb) ([learn more](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)). Alongside the script, there is also the modelling and analysis [script](https://github.com/fulan-m/AnomaFire/blob/main/C_O_M_B_U_S_T_I_O_N-pipeline.ipynb) that culminated on the results of the project.

# Description
## [FIAT sampling](https://github.com/fulan-m/FIAT-FIRMS/blob/main/COMBUSTION/download_training_data.ipynb)
Requires a Google Earth Engine project and a shapefile hosted on the platform to work. Uses string date formats to create monthly tables (YYYY-MM) for each of the four collection types (FIRMS, FIRMS with CHIRPS, VIIRS, and VIIRS with CHIRPS. A example [FIRMS table](https://github.com/fulan-m/FIAT-FIRMS/blob/main/COMBUSTION/firms_monthly_2000_11(in).csv) for November 2000 is available.

## [Main analysis pipeline](https://github.com/fulan-m/FIAT-FIRMS/blob/main/COMBUSTION/C_O_M_B_U_S_T_I_O_N-pipeline.ipynb)
Uses the Scikit-Learn package to train a simple Random Forest algorithm based only on sample rows with complete data (no empty (or NA) data points) for each date throughout the series. Collections with over 50% of NA points are automatically removed from the analysis, where the transformation of a number to a NA can happen when bitmask filtering is applied. After the execution of the main pipeline, '.joblib' models are created for each of the 24 types.

## [Real world testing](https://github.com/fulan-m/FIAT-FIRMS/blob/main/COMBUSTION/test_real_life_scenario.ipynb)
With the resulting '.joblib' files created from the previous script, the creation of a single multi-band image containing all the variables used during training and testing (including bitmasks) is made using a specialized [script](https://github.com/fulan-m/FIAT-FIRMS/blob/main/COMBUSTION/download_multiband_rasters.py). When the models are applied on the image, a FIAT probability estimate map is created.

</div>
