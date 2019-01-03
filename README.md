# COFGA
### Authors: Lukkas Hamman & Rasmus Davidsen

#### This repository contain code and documentation for COFGA project created in the DTU course 02456. The project aims to classify vehicles in aerial photos. It is multilabel fine-grained classification.

#### Contents:
* The folder *results* contains results for the 8 ID's found in the paper. Both csv result files, plots and the python file executed in the DTU HPC cluster is found for all 8 ID's.

* The script *data_utils* is used when padding the images.

* The script *COFGA_dataset.py* is the custom Pytorch dataset class used whn loading and augmenting data. This allows to load data on the fly.

* The script *mAP.py* is used to evaluate the Average Precison (AP) and Mean Average Precision (mAP) of the solution. 

* The notebook *MAFAT_notebook_getting_started_2018-10-12.ipynb* was provided by our superviser Maxim Khomiakov and crops the objects out of the aerial images. Minor changes were applied in order to fit our desired folder structure. 

* The notebook *COFGA_ImagePreprocessing.ipynb* resizes the images to fit Pytorch's ResNet50 and ResNet152 and places these images in a new folder. 

* The notebook *COFGA_LabelPreprocessing.ipynb* preprocesses the labels by using one hot encoding. 

* The notebook *COFGA_main.ipynb* loads and augment the data, constructs the network, train and evaluate the solution for every epoch and save these values to csv files.


