# Gene Expression Clustering - Single Cell RNA Sequencing

Task is to cluster the data into 16 clusters using any of the clustering models. Originally, the dataset comes from Single-cell RNA-seq data. "Single-cell RNA sequencing (scRNA-seq) is a popular and powerful technology that allows you to profile the whole transcriptome of a large number of individual cells."

## Requirements
Jupyter Notebook installation requires Python 3.3 or greater, or Python 2.7, these versions can be found and installed from the official [python website](https://www.python.org/downloads/)

Install Jupyter notebook

```python
python –m pip install –upgrade pip

pip install jupyter
```
## Installation


Execute the below commands in command prompt to install respective libraries used in this code:
```python
pip install pandas
pip install numpy
pip install -U scikit-learn
pip install sklearn
pip install matplotlib
```

## Dataset
We have three files train data and test data and gene names files, “data_tr.txt”, “data_ts.txt” and “gene_names.txt” files. You can download these data files in the [data folder](https://github.com/NarenderKrishna/GeneExpressionClustering/tree/main/data) and code in [py_code](https://github.com/NarenderKrishna/GeneExpressionClustering/tree/main/py_code) folder of project repository. Below are the links for datasets where we can download and use them.

[Training data](https://github.com/NarenderKrishna/GeneExpressionClustering/blob/main/data/trainingdata.zip)

[Test data](https://github.com/NarenderKrishna/GeneExpressionClustering/blob/main/data/testdata.zip)


## Execution
Place the data files “data_tr.txt”, “data_ts.txt “ and “gene_names.txt” files in the respective code location and provide the file path to assigned variables to read the files.

•	Start the Jupyter in the command prompt/terminal, where it opens the localhost in the default web browser. 
```
C:\Users\userfolder >jupyter notebook
```

•	Browse to the folder where code and dataset files are saved and open ```main.py``` file

Run ```main.py``` file which is located in [py_code](https://github.com/NarenderKrishna/GeneExpressionClustering/tree/main/py_code) folder.

•	Press ```Ctrl + Shift + Alt + Enter``` to run the code file.

## Note

Update the working API link in the code file to check the accuracy.


## Advisor and Team

Advisor - Professor Dr. Othman Soufan

Team Members - Madhusree Maddu, Akhil Balla, Narender Krishna Rapolu
