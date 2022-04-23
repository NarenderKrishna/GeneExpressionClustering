import pandas as pd
import seaborn as sb
sb.set()
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import requests


#Drop columns with standard deviation < 0.2    
def drop_columns_std(train_data, test_data):
    labels=train_data.columns
    col_with_std=[]
    for col in labels:
        std_dev = (train_data[col].std() < 0.2)
        if std_dev:
            col_with_std.append(col)
    #print(col_with_std)
    train_data.drop(columns=col_with_std, axis=1, inplace=True)
    test_data.drop(columns=col_with_std, axis=1, inplace=True)
    return train_data, test_data

# Dimensionality reduction
def pca(train_data, test_data):
    pca = PCA(n_components=0.95)
    pca.fit(train_data)
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)
    return train_data_pca, test_data_pca
    
# Applying K-Means clustering algorithm
def kMeans(train_data_pca, test_data_pca):
    model = KMeans(n_clusters=16, random_state=0, init="k-means++")
    model.fit(train_data_pca)
    predictions = model.predict(test_data_pca)
    return predictions

#Silhoutte Score    
def silhoutte_score(test_data, predictions):
    score = silhouette_score(test_data, predictions, metric='euclidean')
    return score

#API Hit and Accuracy prediction
def score(predictions):
    url = "https://www.csci555competition.online/scoretest"
    payload = json.dumps(predictions.tolist())
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text
    
train_data=pd.read_csv('./data_tr.txt',sep="\t",header=None)
gene_data= pd.read_csv('./gene_names.txt', sep="\t",header=None)
test_data = pd.read_csv('./data_ts.txt', sep="\t",header=None)
labels=gene_data[0].tolist()
train_data.columns=labels
test_data.columns=labels
    
train_data, test_data = drop_columns_std(train_data, test_data)
train_data_pca, test_data_pca = pca(train_data, test_data)
kMeans_model = kMeans(train_data_pca, test_data_pca)
silhoutte = silhoutte_score(test_data, kMeans_model)
print("silhoutte score", silhoutte)
accuracy = score(kMeans_model)
print("Accuracy : ", accuracy)
