# INNE
This repository contains the software of the anomaly\outlier detection algorithm INNE (isolation using Nearest Neighbour Ensemble). Refer to the following publications for more technical details about the algorithm.

Conference Paper: Efficient Anomaly Detection by Isolation Using Nearest Neighbour Ensemble (http://ieeexplore.ieee.org/document/7022664)
Thesis: Isolation based anomaly detection: a re-examination(http://arrow.monash.edu.au/vital/access/manager/Repository/monash:162299)

The software is written in JAVA and can be run using the command line (in both Windows and Linux). This software requires JAVA environment to be installed in the system. 

The software accepts the following parameters:
S: Sample Size (default value 8)
T: Ensemble Size (default value 100)
dataFile: dataset name 
fileType: dataset format (this software only accepts ARFF and CSV formats. CSV dataset must contain a header row as the first row)
hasLabels: set this flag if the dataset has labels. The labels must be 0 (normal\inlier) and 1 (anomaly\outlier). Labels must be in the last column of the dataset. If this flag is set then the software will calculate the AUC (Area Under the Receiver Operating Characteristic curve) value of the anomaly detector.
 
Example command line call: java -cp INNE.jar outlier.INNE -S 32 -T 100 -dataFile sample_data.arff -hasLabels -fileType arff
