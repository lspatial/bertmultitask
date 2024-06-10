import torch
import os
import re
import pandas as pd
import shutil 
import random, numpy as np, argparse
import zipfile


def zip_dir(folder_path, output_path):
    """
    Compress the specified folder into a zip file.

    :param folder_path: Path to the folder to compress.
    :param output_path: Path where the zip file will be saved.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # The arcname argument ensures the files are stored with the correct relative paths
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                
class EnsembleBertModel:

    def __init__(self, threshold = 0.7):
        self.threshold = threshold
        self.statfile = '/adt_geocomplex/bert_allresult/bertallres_'+str(threshold) + '.csv'

    def retrieveValidPredictions(self,root):
        retrResults=[]
        folders = [folder for folder, d_names, f_names in os.walk(root) if os.path.isdir(folder)]
        folders = [folder for folder in folders if os.path.basename(folder).startswith('multitasks.')]

        for folder in folders:
            predFolders = [afld for afld, d_names, f_names in os.walk(folder) if os.path.isdir(afld) and os.path.basename(afld).isdigit()]
            for apredfld in predFolders:
               apredictiondir = apredfld + '/predictions'
               aperformance = apredfld + '/devacc_combined.csv'
               if not os.path.exists(apredictiondir) or not os.path.exists(aperformance):
                    continue
               perfDf = pd.read_csv(aperformance)
               if perfDf.size < 1 or perfDf.empty:
                    continue
               if perfDf['total'].isnull().any():
                    continue
               perfDf['total_up'] = (perfDf['sst']+perfDf['para']+(perfDf['sts']+1.0)/2.0)/3.0
               maxindex=np.argmax(perfDf['total_up'])
               AResult = pd.DataFrame({'modelconfig': os.path.basename(folder), 
                         'subnum': os.path.basename(apredfld), 'total_up':max(perfDf['total_up']),
                         'maxdevtotal':max(perfDf['total_up']),
                         'sst_acc': perfDf.iloc[maxindex]['sst'], 
                         'para_acc':perfDf.iloc[maxindex]['para'],
                         'sts_cor': perfDf.iloc[maxindex]['sts'],
                         'dir':folder},index=[0])
               retrResults.append(AResult)
        retrieveResults=pd.concat(retrResults)
        retrieveResults.to_csv(self.statfile,index=False)
        return retrieveResults 
      
      
    def readAPrediction(self, pmetrics,type='dev',task='sst'):
        suffix = task +'-'+ type + '-output.csv'
        allresults=[] 
        for i in range(pmetrics.shape[0]):
            afdevpmetric = pmetrics.iloc[i]
            prepath = afdevpmetric['dir'] + '/' + str(afdevpmetric['subnum']) + '/predictions/' + suffix
            if os.path.exists(prepath):
              aresult = pd.read_csv(prepath,names=['id','target'],skiprows=1,sep=',')
              if task == 'sst':
                 aresult.columns = ['id', 'Predicted_Sentiment'] 
              elif task == 'para':
                 aresult.columns = ['id', 'Predicted_Is_Paraphrase'] 
              elif task == 'sts':
                 aresult.columns = ['id', 'Predicted_Similiary'] 
              allresults.append(aresult)
        allresults = pd.concat(allresults)
        print('nrow(allresults):',allresults.shape) 
        print('allresults columns:',allresults.columns) 
        if task == 'sst':
          finalRes = allresults.groupby('id')['Predicted_Sentiment'].apply(lambda x: x.mode().iloc[0])
        elif task == 'para':
          finalRes = allresults.groupby('id')['Predicted_Is_Paraphrase'].apply(lambda x: x.mode().iloc[0])
        elif task == 'sts': 
          finalRes = allresults.groupby('id')['Predicted_Similiary'].mean()
        return  finalRes 
        
      
    def ensPredictions(self,type='dev'):
        devpmetrics = pd.read_csv(self.statfile)
        devpmetrics = devpmetrics[devpmetrics['total_up']>=self.threshold]
        print(' valid data count:',devpmetrics.shape[0])
        sst_dev_result=self.readAPrediction(devpmetrics,type=type,task='sst')
        para_dev_result=self.readAPrediction(devpmetrics,type=type,task='para')
        sts_dev_result=self.readAPrediction(devpmetrics,type=type,task='sts')
        outroot = '/adt_geocomplex/bert_allresult/ensemble_up/prop_'+str(self.threshold)
        devroot = outroot+'/' + type + '/predictions'        
        if not os.path.exists(devroot):
           os.makedirs(devroot) 
        targetf = devroot+'/sst-' + type + '-output.csv'
        sst_dev_result.to_csv(targetf,index=True,index_label='id')
        targetf = devroot+'/para-' + type + '-output.csv'
        para_dev_result.to_csv(targetf,index=True,index_label='id')
        targetf = devroot+'/sts-' + type + '-output.csv'
        sts_dev_result.to_csv(targetf,index=True,index_label='id')
        output_zip_file = outroot + '/' + type +'/'+ type +'_predictions.zip' 
        zip_dir(devroot, output_zip_file) 
  
  

if __name__ == '__main__':
    props = [0.65,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80]
    props = [0.78,0.77,0.76]
    props = [0.8]
    for p in props: 
        print(p,' ... ...')
        myensemble = EnsembleBertModel(p) 
        allindresults = myensemble.retrieveValidPredictions(root='/adt_geocomplex/bert_allresult') 
        print(len(allindresults))
        myensemble.ensPredictions(type='dev') 
        myensemble.ensPredictions(type='test')  









