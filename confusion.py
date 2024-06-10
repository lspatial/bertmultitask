import pandas as pd
import csv
from sklearn import metrics

import seaborn as sns
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as plticker
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
def readDevData(filename):
    num_labels = {}
    data = []
    with open(filename, 'r', encoding='utf-8-sig') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            sent_id = record['id'].lower().strip()
            label = int(record['sentiment'].strip())
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            data.append((sent, label, sent_id))
    obsData = pd.DataFrame({'sent':[r[0] for r in data],
                            'label':[r[1] for r in data],
                            'id':[r[2] for r in data]})
    return obsData

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

def readParaData(filename):
    num_labels = {}
    data = []
    with open(filename, 'r', encoding="utf-8-sig") as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            try:
                sent_id = record['id'].lower().strip()
                data.append((preprocess_string(record['sentence1']),
                             preprocess_string(record['sentence2']),
                             int(float(record['is_duplicate'])), sent_id))
            except:
                pass
    obsData = pd.DataFrame({'id':[r[3] for r in data],
                            'sentence1':[r[0] for r in data],
                            'sentence2': [r[1] for r in data],
                            'is_duplicate':[r[2] for r in data]})
    return obsData

def readStsData(filename):
    num_labels = {}
    data = []
    with open(filename, 'r', encoding="utf-8-sig") as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            try:
                sent_id = record['id'].lower().strip()
                data.append((preprocess_string(record['sentence1']),
                             preprocess_string(record['sentence2']),
                             int(float(record['similarity'])), sent_id))
            except:
                pass
    obsData = pd.DataFrame({'id':[r[3] for r in data],
                            'sentence1':[r[0] for r in data],
                            'sentence2': [r[1] for r in data],
                            'similarity':[r[2] for r in data]})
    return obsData


def readSSTData():
    tfl='D:/wkspacebug/nlp_G_project/finaprediction/prop_0.78/dev/predictions/sst-dev-output.csv'
    resdata=pd.read_csv(tfl,names=['id','target'],skiprows=1,sep=',')
    resdata['id'] = resdata['id'].str.strip().str.lower()
    tfl='D:/wkspacebug/nlp_G_project/bertdprof2024s/data/ids-sst-dev.csv'
    targetdata=readDevData(tfl)
    targetdata=targetdata.merge(resdata,how='inner',on='id')
    targetdata['residual']=targetdata['label']-targetdata['target']
    outfl = 'D:/wkspacebug/nlp_G_project/finaprediction/merged_sst-dev.csv'
    targetdata.to_csv(outfl,index=False)
    return targetdata


def readPARAData():
    tfl='D:/wkspacebug/nlp_G_project/finaprediction/prop_0.78/dev/predictions/para-dev-output.csv'
    resdata=pd.read_csv(tfl,names=['id','target'],skiprows=1,sep=',')
    resdata['id'] = resdata['id'].str.strip().str.lower()
    tfl='D:/wkspacebug/nlp_G_project/bertdprof2024s/data/quora-dev.csv'
    targetdata=readParaData(tfl)
    targetdata=targetdata.merge(resdata,how='inner',on='id')
    outfl = 'D:/wkspacebug/nlp_G_project/finaprediction/merged_para-dev.csv'
    targetdata.to_csv(outfl,index=False)
    return targetdata

def readSTSData():
    tfl='D:/wkspacebug/nlp_G_project/finaprediction/prop_0.78/dev/predictions/sts-dev-output.csv'
    resdata=pd.read_csv(tfl,names=['id','target'],skiprows=1,sep=',')
    resdata['id'] = resdata['id'].str.strip().str.lower()
    tfl='D:/wkspacebug/nlp_G_project/bertdprof2024s/data/sts-dev.csv'
    targetdata=readStsData(tfl)
    targetdata=targetdata.merge(resdata,how='inner',on='id')
    targetdata['dif']=targetdata['similarity']-targetdata['target']
    outfl = 'D:/wkspacebug/nlp_G_project/finaprediction/merged_sts-dev.csv'
    targetdata.to_csv(outfl,index=False)
    return targetdata


def retrieveConfMatrix():
    tarData = readSSTData()
    categories = ['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive']
    conf_matrix = metrics.confusion_matrix(tarData['label'].values, tarData['target'].values, labels=[0, 1, 2, 3, 4])

def confusionSKPlot():
    tarData = readSSTData()
    categories = ['negative', 'somewhat \n negative', 'neutral', 'somewhat \n positive', 'positive']
    conf_matrix = metrics.confusion_matrix(tarData['label'].values, tarData['target'].values, labels=[0,1,2,3,4])

    plt.figure(figsize=(6, 6))
    #sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=plt.cm.Reds)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=plt.cm.Reds) #, xticklabels=categories, yticklabels=categories)
    plt.xticks(np.arange(5) + 0.5, categories, rotation=45)
    plt.yticks(np.arange(5) + 0.5, categories, rotation=0)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.show()
    plt.draw()
    plt.savefig('D:/wkspacebug/nlp_G_project/finareport/figs/senti_cls.png',dpi=100,bbox_inches='tight')

def confusionParaPlot():
    tarData = readPARAData()
    categories = ['no \n paraphrase', 'paraphrase']
    conf_matrix = metrics.confusion_matrix(tarData['is_duplicate'].values, tarData['target'].values, labels=[0,1])

    plt.figure(figsize=(6, 6))
    #sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=plt.cm.Reds)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=plt.cm.Reds) #, xticklabels=categories, yticklabels=categories)
    plt.xticks(np.arange(2) + 0.5, categories, rotation=45)
    plt.yticks(np.arange(2) + 0.5, categories, rotation=0)
    plt.xlabel('Predicted Paraphrase', fontsize=12)
    plt.ylabel('Actual Paraphrase', fontsize=12)
    plt.show()
    plt.draw()
    plt.savefig('D:/wkspacebug/nlp_G_project/finareport/figs/para_cls.png',dpi=100,bbox_inches='tight')


def scatterSTSPlot():
    tarData = readSTSData()
    x=tarData['similarity'].values
    y=tarData['target'].values
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)
    plt.show()
    plt.draw()
    plt.savefig('D:/wkspacebug/nlp_G_project/finareport/figs/sts_cls.png',dpi=100,bbox_inches='tight')



if __name__ == '__main__':
    scatterSTSPlot()
    #confusionParaPlot()

