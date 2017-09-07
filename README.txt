All of the code to produce the numerical values reported in the analysis are available in these files. This is research code and may require more detailed assistance to be fully functional.

Two directories contain the code to download and process the crop (crop_data) and weather station (weather_data) used in the analysis.

The model folder contains the design_mat.py script which aggregates these data into design matrices for use in the regression results reported. The tr_models.py script conducts the analysis and figs.py produces the figures. Several figures require variables created during tr_models.

Note that an up to date US census file is needed from https://catalog.data.gov/dataset/tiger-line-shapefile-2016-nation-u-s-current-county-and-equivalent-national-shapefile
