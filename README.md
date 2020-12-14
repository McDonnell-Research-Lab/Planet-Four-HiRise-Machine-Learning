# HiRISE-MachineLearning-Python
 Using deep neural networks to automatically find fans in HiRISE polar imagery.
 
 Code in association with manuscript: "A Neural Network's Search For Polar Spring-time Fans On Mars."
 
 It is recommended to run all cells in notebooks in this order:
 
 - Notebooks/DownloadHiRISE/DownloadHiRISE.ipynb
 - Notebooks/SemanticSegmenterCNN/HiRISE_Segmenter_Train.ipynb (set Preload to False)
 - Notebooks/SemanticSegmenterCNN/HiRISE_Segmenter_ValidationResults.ipynb
 - Notebooks/TileClassifierCNN/HiRISE_tile_classifier_train_and_val.ipynb (set Preload to True if HiRISE_Segmenter_Train.ipynb ran first)
 - Notebooks/CreateFigures/Figures_6_through_12.ipynb 
 - Notebooks/CreateFigures/Figures13_and_14.ipynb

The notebook Notebooks/Clustering/HiRISE_ClusteringResults.ipynb is included for completeness, but cannot be run, as it relies on many large mask images created in proprietary software.  The results of running this script are however stored in Data/ClusteringResults/ which enables Notebooks/CreateFigures/Figures_6_through_12.ipynb to be run once the training and validation notebooks are run.

Note that our training scripts use over 300 GB of RAM, and the segmentation one uses 4 GPUs in parallel. To run without these resources, changes are need, as indicated in the notebooks.
 
 
