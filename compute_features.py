import os
import numpy as np
import pandas as pd
import warnings
import logging
import datetime
import calendar
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')


logging.basicConfig(filename = 'file.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(name='mylogger')


def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.datetime.utcfromtimestamp(timestamp)


def cpte_chq_feature(read_path: str, file_save_path: str) -> None:
    if not os.path.isdir(file_save_path):
        os.mkdir(file_save_path)

    cpte_features = pd.DataFrame()
    for f in os.listdir(read_path):
        aux = pd.read_csv(read_path + '/' + f,
                          parse_dates=['Start_Date', 'End_Date_Rebuild', 'New_Start_Date'])
        
        #aux = aux[aux['Party_Id'] !='IA0000000']
        aux= aux[(aux["Party_Id"]!="?") & (aux["Party_Id"]!=" ") & (aux["Party_Id"]!="000000000") & (aux.Party_Id!='IA0000000')]
        aux['New_End_Date']=  aux['New_End_Date'].astype('string')
        aux['Party_Id']=  aux['Party_Id'].astype('float')
        aux['New_End_Date'] = pd.to_datetime(aux['New_End_Date'],errors="coerce", format='%d.%m.%Y')
        aux.loc[(aux['New_End_Date'].isnull()), 'New_End_Date'] = datetime.datetime(aux.year.unique()[0], aux.month.unique()[0],
                              calendar.monthrange(aux.year.unique()[0], aux.month.unique()[0])[1])

        aux['Account_Balance'] = aux['Account_Balance'].astype(str).str.split(',', expand=True)[0].astype(np.int64)
        solde_fin_mois = aux.sort_values(by=['Start_Date'], ascending=False)
        solde_fin_mois.drop_duplicates(subset=['Party_Id', 'Account_Id', 'Host_Account_Nbr'], keep='first',
                                       inplace=True)
        #, 'New_End_Date'
        solde_fin_mois = solde_fin_mois.groupby(['Party_Id', 'year', 'month'])['Account_Balance'].sum().reset_index()
        solde_fin_mois.rename(index=str, columns={'Account_Balance': 'solde_fin_mois_cav'}, inplace=True)
        solde_debiteur = aux.loc[aux['Account_Balance'] < 0]
        solde_debiteur['nbre_jour_debit'] = solde_debiteur['New_End_Date'] - solde_debiteur['Start_Date']
        solde_debiteur['nbre_jour_debit'] = solde_debiteur['nbre_jour_debit'].dt.days
        
        nbre_jour_debit = solde_debiteur.groupby(['Party_Id', 'year', 'month'])['nbre_jour_debit'].max().reset_index()
         
        nbre_fois_debit = solde_debiteur.groupby(['Party_Id', 'year', 'month']).size().reset_index().rename(index=str, columns={0: 'nbre_fois_debit'})

        frame = solde_fin_mois.merge(nbre_jour_debit, on=['Party_Id', 'year', 'month'],
                                                  how='left')
        frame = frame.merge(nbre_fois_debit, on=['Party_Id', 'year', 'month'],
                                                      how='left')
        
        frame.fillna(0, inplace=True)

        del solde_fin_mois, nbre_jour_debit, nbre_fois_debit,aux
        #print("########################")
        #print(frame[(frame['Party_Id']==11.0) & (frame['year']==2019) & (frame['month']==8)])
        
        cpte_features=pd.concat([cpte_features,frame0])
        #print(cpte_features[(cpte_features['Party_Id']==11.0) & (cpte_features['year']==2019) & (cpte_features['month']==8)])
        #rint("########################")

        del frame
    cpte_features.to_csv(file_save_path+'/cpte_cheq_features.csv',index=False)





def cpte_epargne_feature(read_path: str, file_save_path: str) -> None:
    k=0
    if not os.path.isdir(file_save_path):
        os.mkdir(file_save_path)

    cpte_features = pd.DataFrame()
    for f in os.listdir(read_path):
        aux = pd.read_csv(read_path + '/' + f,
                          parse_dates=['Start_Date', 'End_Date_Rebuild', 'New_Start_Date', 'New_End_Date'])
        
        aux = aux[aux['Party_Id'] !='IA0000000']
        aux['Account_Balance'] = aux['Account_Balance'].astype(str).str.split(',', expand=True)[0].astype(np.int64)
        solde_fin_mois = aux.sort_values(by=['Start_Date'], ascending=False)
        solde_fin_mois.drop_duplicates(subset=['Party_Id', 'Account_Id', 'Host_Account_Nbr'], keep='first',
                                       inplace=True)
        #, 'New_End_Date'
        #solde_fin_mois = solde_fin_mois.groupby(['Party_Id', 'year', 'month'])['Account_Balance'].sum().reset_index()
         #solde_fin_mois.rename(index=str, columns={'Account_Balance': 'solde_fin_mois_epargne'}, inplace=True)
       
    #me debut
        solde_fin_mois =solde_fin_mois.groupby(['Party_Id', 'year', 'month']).agg({'Account_Balance': ['sum','median','mean','min', 'max']})  
    # rename columns
        solde_fin_mois.columns = ['solde_fin_mois_epargne', 'median_fin_mois_epargne', 'mean_fin_mois_epargne','min_fin_mois_epargne','max_fin_mois_epargne']
    # reset index to get grouped columns back
        solde_fin_mois = solde_fin_mois.reset_index()
    #### me fin
    
       

        del aux


        cpte_features=pd.concat([cpte_features,solde_fin_mois])
        k+=1

        logger.info(k)
        logger.info(cpte_features.shape)


        #print(cpte_features.shape)

        del solde_fin_mois


        '''solde_fin_mois.to_csv(file_save_path+'/cpte_epargne_features_{}_{}.csv'.format(f.split('_')[-2],
                                                                                  f.split('_')[-1]), index=False)'''

    cpte_features.to_csv(file_save_path + '/cpte_epargne_features.csv', index=False)


def impaye_feature(read_path: str, file_save_path: str) -> None:

    if not os.path.isdir(file_save_path):
        os.mkdir(file_save_path)


    for f in os.listdir(read_path):
        impaye_features = pd.read_csv(read_path + '/' + f,
                          parse_dates=['Start_Date', 'Repymt_Date', 'New_Start_Date', 'New_End_Date'])

        p=impaye_features['New_End_Date'].unique()[0]
        d_date = to_datetime(p)
        
        #
        past_3month = d_date - relativedelta(months=3)
        past_6month = d_date - relativedelta(months=6)
        past_12month = d_date - relativedelta(months=12)

        impayes_3month = impaye_features.loc[(impaye_features['Repymt_Status'] == 8) & (impaye_features['Start_Date'].dt.date > past_3month.date())]
        impayes_6month = impaye_features.loc[(impaye_features['Repymt_Status'] == 8) & (impaye_features['Start_Date'].dt.date > past_6month.date())]
        impayes_12month = impaye_features.loc[(impaye_features['Repymt_Status'] == 8) & (impaye_features['Start_Date'].dt.date > past_12month.date())]

        impayes_3month.drop_duplicates(subset=['Agreement_Id', 'Party_Id', 'Repymt_Nbr'], inplace=True)
        impayes_3month = impayes_3month.groupby(['Party_Id']).size().reset_index().rename(index=str,
                                                                                          columns={0: 'nbr_impayes_3month'})

        impayes_6month.drop_duplicates(subset=['Agreement_Id', 'Party_Id', 'Repymt_Nbr'], inplace=True)
        impayes_6month = impayes_6month.groupby(['Party_Id']).size().reset_index().rename(index=str,
                                                                                          columns={0: 'nbr_impayes_6month'})

        impayes_12month.drop_duplicates(subset=['Agreement_Id', 'Party_Id', 'Repymt_Nbr'], inplace=True)
        impayes_12month = impayes_12month.groupby(['Party_Id']).size().reset_index().rename(index=str, columns={
            0: 'nbr_impayes_12month'})

        impaye_en_cours = impaye_features.groupby(['Agreement_Id', 'Party_Id', 'Repymt_Nbr', 'Repymt_Date'])[
            'Repymt_Status'].max().reset_index()
        impaye_en_cours = impaye_en_cours.loc[(impaye_en_cours['Repymt_Status'] == 8)]
        impaye_en_cours = impaye_en_cours.groupby(['Party_Id'])['Repymt_Date'].max().reset_index()
        impaye_en_cours['Repymt_Date'] = pd.to_datetime(impaye_en_cours['Repymt_Date'])
        impaye_en_cours['Nbre_jours_retard'] = d_date.date() - impaye_en_cours['Repymt_Date'].dt.date
        impaye_en_cours['Nbre_jours_retard'] = impaye_en_cours['Nbre_jours_retard'].dt.days



        impaye_features = impayes_3month.merge(impayes_6month, on=['Party_Id'], how='outer')
        impaye_features = impaye_features.merge(impayes_12month,  on=['Party_Id'], how='outer')
        impaye_features = impaye_features.merge(impaye_en_cours,  on=['Party_Id'], how='outer')

        r1=f.split("_")[2:]
        r1 = "_".join(r1)
        impaye_features['year'] = r1.split("_")[1].split(".")[0]
        impaye_features['month'] = r1.split("_")[0]

        impaye_features.to_csv(file_save_path +'/'+ f, index=False)


def get_all_file(read_path:str,save_path:str)->None:

    data = pd.DataFrame()
    for f in os.listdir(read_path):
        aux = pd.read_csv(read_path + '/' + f)
        data = pd.concat([data, aux])

    data.to_csv(save_path,index=False)


def add_cli(read_path:str , save_path:str)->None:
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    cpte = pd.read_csv('../data/compte_cheque.txt', usecols=['Party_Id', 'Host_Account_Nbr'], sep=';')
    cpte.drop_duplicates(inplace=True)
    cpte['Party_Id'] = cpte['Party_Id'].astype(str).str.strip()
    cpte['Host_Account_Nbr'] = cpte['Host_Account_Nbr'].astype(int)
    for f in os.listdir(read_path):
        aux = pd.read_csv(read_path + '/' + f)
        aux['account_id1'] = aux['account_id1'].astype(int)
        aux = aux.merge(cpte,left_on=['account_id1'], right_on=['Host_Account_Nbr'])
        aux.to_csv(save_path+'/'+f, index=False)

    
