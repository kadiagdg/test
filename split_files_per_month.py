import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import calendar
import numpy as np






def split_files(read_path: str, save_path: str, file_name: str, date_columns: str, chunk_size: int,
                col_to_convert: str) -> None:
    """
    :param read_path: file to read . This file will be split per month
    :param save_path: folder where to save splitted files
    :param file_name: file name
    :param date_columns: column date according to which to make the split
    :param chunk_size: Number of row in the chunk
    :param col_to_convert: column to convert
    :return: None
    """
    if not os.path.exists(save_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(save_path)
    nb_rows = chunk_size

    for chunk in pd.read_csv(read_path, sep=';', chunksize=nb_rows,
                             na_values=['', ' ', ' ' * 2, ' ' * 3, ' ' * 4, ' ' * 5, '?']):

        chunk[date_columns] = pd.to_datetime(chunk[date_columns], format='%d.%m.%Y')
        chunk['year'] = chunk[date_columns].dt.year
        chunk['month'] = chunk[date_columns].dt.month
        period = chunk.drop_duplicates(subset=['year', 'month'])[['year', 'month']].values
        for p in period:
            new = chunk.loc[(chunk['year'] == p[0]) & (chunk['month'] == p[1])]
            try:
                new[col_to_convert] = new[col_to_convert].astype(str).str.strip().str.replace(',', '')
                new[col_to_convert] = new[col_to_convert].astype(int)
            except:
                pass

            if file_name + '_{}_{}.csv'.format(p[1], p[0]) in os.listdir(save_path):
                old = pd.read_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]))
                new = pd.concat([old, new])
            new.to_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]), index=False)

        """ Script modifie par Essan """


def split_files2(read_path: str, save_path: str, file_name: str, date_columns: str, chunk_size: int,
                 col_to_convert: str) -> None:
    if not os.path.exists(save_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(save_path)
    nb_rows = chunk_size

    for chunk in pd.read_csv(read_path, sep=';', chunksize=nb_rows,
                             na_values=['', ' ', ' ' * 2, ' ' * 3, ' ' * 4, ' ' * 5, '?']):

        chunk[date_columns] = pd.to_datetime(chunk[date_columns], format='%d.%m.%Y')
        chunk['year'] = chunk[date_columns].dt.year
        chunk['month'] = chunk[date_columns].dt.month
        period = chunk.drop_duplicates(subset=['year', 'month'])[['year', 'month']].values

        for p in period:
            new = chunk.loc[(chunk['year'] == p.astype(int)[0]) & (chunk['month'] == p.astype(int)[1])]
            # print(p.astype(int))
            try:
                new[col_to_convert] = new[col_to_convert].astype(str).str.strip().str.replace(',', '')
                new[col_to_convert] = new[col_to_convert].astype(int)
                # print(new)
            except:
                pass

            if file_name + '_{}_{}.csv'.format(p.astype(int)[1], p.astype(int)[0]) in os.listdir(save_path):
                old = pd.read_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p.astype(int)[1], p.astype(int)[0]))
                new = pd.concat([old, new])
            new.to_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p.astype(int)[1], p.astype(int)[0]),
                       index=False)


def split_files_for_cpte_cheque(read_path: str, save_path: str, file_name: str, date_columns: str, chunk_size: int,
                                col_to_convert: str) -> None:
    """
    :param read_path: file to read . This file will be split per month
    :param save_path: folder where to save splitted files
    :param file_name: file name
    :param date_columns: column date according to which to make the split
    :param chunk_size: Number of row in the chunk
    :param col_to_convert: column to convert
    :return: None
    """
    if not os.path.exists(save_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(save_path)
    nb_rows = chunk_size

    for chunk in pd.read_csv(read_path, sep=';', chunksize=nb_rows,
                             na_values=['', ' ', ' ' * 2, ' ' * 3, ' ' * 4, ' ' * 5, '?']):
        chunk[date_columns] = pd.to_datetime(chunk[date_columns], format='%d.%m.%Y')
        chunk['Start_Date'] = pd.to_datetime(chunk['Start_Date'], format='%d.%m.%Y')
        chunk['End_Date_Rebuild'] = pd.to_datetime(chunk['End_Date'], format='%d.%m.%Y', errors='coerce')
        # this is to deal with the limit of pandas datetime
        chunk['End_Date_Rebuild'] = chunk['End_Date_Rebuild'].fillna(
            datetime.datetime.today().date() + relativedelta(years=40))

        chunk['year'] = chunk[date_columns].dt.year
        chunk['month'] = chunk[date_columns].dt.month
        period = chunk.drop_duplicates(subset=['year', 'month'])[['year', 'month']]
        #period = pd.concat([period,pd.DataFrame({'year':[2022],'month':[3]})])
        period['end_day_of_month'] = period.apply(lambda x: calendar.monthrange(x['year'], x['month'])[1],
                                                  axis=1).values
        aux = pd.DataFrame({'year': period['year'].values, 'month': period['month'].values,
                            'day': period['end_day_of_month'].values})
        period['end_day_of_month'] = pd.to_datetime(aux).values
        del aux

        aux = pd.DataFrame({'year': period['year'].values, 'month': period['month'].values,
                            'day': [1] * period.shape[0]})

        period['start_day_of_month'] = pd.to_datetime(aux).values

        # period=[['year', 'month']].values
        del aux
        period = period[['year', 'month', 'start_day_of_month', 'end_day_of_month']].values
        chunk['End_Date_Rebuild'] = pd.to_datetime(chunk['End_Date_Rebuild']).values
        #print(period)
        for p in period:
            # new = chunk.loc[(chunk['year'] == p[0]) & (chunk['month'] == p[1])]
            new = chunk.loc[((chunk['year'] == p[0]) & (chunk['month'] == p[1])) | (
                    (chunk['End_Date_Rebuild'].dt.year == p[0]) & (chunk['End_Date_Rebuild'].dt.month == p[1])) | (
                                    (chunk['Start_Date'] <= p[3]) & (chunk['End_Date_Rebuild'] >= p[3]))]

            new['year'] = p[0]
            new['month'] = p[1]
            new['New_Start_Date'] = new['Start_Date'].values
            new['New_End_Date'] = new['End_Date'].values
            new.loc[(new['Start_Date'].dt.month != new['month']) | (
                    new['Start_Date'].dt.year != new['year']), 'New_Start_Date'] = datetime.datetime(p[0], p[1], 1)

            new.loc[(new['End_Date_Rebuild'].dt.month != new['month']) | (
                    new['End_Date_Rebuild'].dt.year != new['year']), 'New_End_Date'] = datetime.datetime(p[0], p[1],
                      calendar.monthrange(p[0],p[1])[1])
            try:
                new[col_to_convert] = new[col_to_convert].astype(str).str.strip().str.replace(',', '')
                new[col_to_convert] = new[col_to_convert].astype(int)
            except:
                pass

            if file_name + '_{}_{}.csv'.format(p[1], p[0]) in os.listdir(save_path):
                old = pd.read_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]))
                new = pd.concat([old, new])
            new.to_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]), index=False)

                


def split_files_for_h_overdraft(read_path: str, save_path: str, file_name: str, chunk_size: int,
                                col_to_convert: str) -> None:
    if not os.path.exists(save_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(save_path)
    nb_rows = chunk_size

    dates = os.listdir(save_path)
    dates = [c.split('_')[3] + '_' + c.split('_')[4].split('.')[0] for c in dates]
    print(dates)

    for chunk in pd.read_csv(read_path, sep=';', chunksize=nb_rows,
                             na_values=['', ' ', ' ' * 2, ' ' * 3, ' ' * 4, ' ' * 5, '?']):
        chunk['Contract_Date'] = pd.to_datetime(chunk['Contract_Date'], format='%d.%m.%Y')
        chunk['Overdraft_End_Date'] = pd.to_datetime(chunk['Overdraft_End_Date'], format='%d.%m.%Y', errors='coerce')
        chunk['Maturity_Date'] = pd.to_datetime(chunk['Maturity_Date'], format='%d.%m.%Y', errors='coerce')
        chunk['min_end'] = pd.to_datetime(chunk['min_end'], format='%d.%m.%Y', errors='coerce')
        chunk.loc[chunk['min_end'].isnull(), 'min_end'] = chunk.loc[
            chunk['min_end'].isnull(), 'Overdraft_End_Date'].values
        chunk['Start_Date'] = pd.to_datetime(chunk['Start_Date'], format='%d.%m.%Y')
        chunk['End_Date_Rebuild'] = pd.to_datetime(chunk['End_Date'], format='%d.%m.%Y', errors='coerce')
        # this is to deal with the limit of pandas datetime
        chunk['End_Date_Rebuild'] = chunk['End_Date_Rebuild'].fillna(
            datetime.datetime.today().date() + relativedelta(years=40))

        chunk['Overdraft_End_Date'] = chunk['Overdraft_End_Date'].fillna(
            datetime.datetime.today().date() + relativedelta(years=40))

        chunk['Maturity_Date'] = chunk['Maturity_Date'].fillna(
            datetime.datetime.today().date() + relativedelta(years=40))

        chunk['min_end'] = pd.to_datetime(chunk['min_end'], format='%d.%m.%Y', errors='coerce')

        chunk['year'] = chunk['Start_Date'].dt.year
        chunk['month'] = chunk['Start_Date'].dt.month
        period = chunk.drop_duplicates(subset=['year', 'month'])[['year', 'month']]
        period['end_day_of_month'] = period.apply(lambda x: calendar.monthrange(x['year'], x['month'])[1],
                                                  axis=1).values
        aux = pd.DataFrame({'year': period['year'].values, 'month': period['month'].values,
                            'day': period['end_day_of_month'].values})
        period['end_day_of_month'] = pd.to_datetime(aux).values
        del aux

        aux = pd.DataFrame({'year': period['year'].values, 'month': period['month'].values,
                            'day': [1] * period.shape[0]})

        period['start_day_of_month'] = pd.to_datetime(aux).values
        del aux
        period['m_y'] = period['month'].astype(str) + '_' + period['year'].astype(str)
        period = period.loc[~period['m_y'].isin(dates)]
        period.drop(['m_y'], axis=1, inplace=True)
        chunk['End_Date_Rebuild'] = pd.to_datetime(chunk['End_Date_Rebuild']).values
        chunk['m_y'] = chunk['month'].astype(str) + '_' + chunk['year'].astype(str)
        chunk = chunk.loc[~chunk['m_y'].isin(dates)]
        chunk.drop(['m_y'], axis=1, inplace=True)
        for p in period.values:
            # new = chunk.loc[(chunk['year'] == p[0]) & (chunk['month'] == p[1])]
            new = chunk.loc[((chunk['year'] == p[0]) & (chunk['month'] == p[1])) | (
                    (chunk['min_end'].dt.year == p[0]) & (chunk['min_end'].dt.month == p[1])) | (
                                    (chunk['Contract_Date'] <= p[3]) & (chunk['min_end'] >= p[3]))]

            try:
                new[col_to_convert] = new[col_to_convert].astype(str).str.strip().str.replace(',', '')
                new[col_to_convert] = new[col_to_convert].astype(int)
            except:
                pass

            # new = new.loc[(new['year'] == p[0]) & (new['month'] == p[1])]
            new = new.loc[new['Contract_Status'] == 2]

            new['Agreement_Balance'] = new['Agreement_Balance'].astype(str).str.split(',', expand=True)[0].astype(float)
            new['Overdraft_Limit_Amt'] = new['Overdraft_Limit_Amt'].astype(str).str.split(',', expand=True)[0].astype(
                float)

            nb_jour_debit_non_auto = new.loc[(new['Agreement_Balance'] < 0)]
            nb_jour_debit_non_auto = nb_jour_debit_non_auto.loc[(nb_jour_debit_non_auto['Agreement_Balance'] +
                                                                 nb_jour_debit_non_auto['Overdraft_Limit_Amt']) < 0]

            nb_jour_debit_non_auto['nb_jour_debit_non_auto'] = nb_jour_debit_non_auto['End_Date_Rebuild'].dt.date - \
                                                               nb_jour_debit_non_auto['Start_Date'].dt.date

            nb_jour_debit_non_auto['nb_jour_debit_non_auto'] = nb_jour_debit_non_auto['nb_jour_debit_non_auto'].dt.days

            nb_fois_debit_non_auto = nb_jour_debit_non_auto.groupby(
                ['Host_Agreement_Id_2']).size().reset_index().rename(index=str,
                                                                     columns={0: 'nb_fois_debit_non_auto'})

            nb_jour_debit_non_auto = nb_jour_debit_non_auto.groupby(['Host_Agreement_Id_2'])[
                'nb_jour_debit_non_auto'].sum().reset_index()

            nb_jour_debit_non_auto = nb_jour_debit_non_auto.merge(nb_fois_debit_non_auto, on='Host_Agreement_Id_2')

            del nb_fois_debit_non_auto

            decouv = new.drop_duplicates(subset=['Host_Agreement_Id_2', 'Party_Id', 'Contract_Date', 'min_end'])
            decouv = decouv[['Host_Agreement_Id_2', 'Party_Id', 'Contract_Date', 'min_end', 'Overdraft_Limit_Amt']]
            decouv['Host_Agreement_Id_2'] = decouv['Host_Agreement_Id_2'].astype(int)

            del new
            try:
                dm_bank_account = pd.read_csv('../data/dm_bank_acount/dm_bank_acount_{}_{}.csv'.format(p[1], p[0]),
                                              parse_dates=['Date_Valid'])
            except:
                continue
            dm_bank_account['Account_Balance'] = \
            dm_bank_account['Account_Balance'].astype(str).str.split(',', expand=True)[
                0].astype(float)
            dm_bank_account['account_id1'] = dm_bank_account['account_id1'].astype(int)

            dm_bank_account = dm_bank_account.merge(decouv, left_on=['account_id1'], right_on=['Host_Agreement_Id_2'],
                                                    how='left')

            del decouv

            dm_bank_account['Overdraft_Limit_Amt'] = dm_bank_account['Overdraft_Limit_Amt'].fillna(0)
            dm_bank_account['Montant_decouvert_dispo'] = dm_bank_account['Overdraft_Limit_Amt'].values
            dm_bank_account.loc[dm_bank_account['Account_Balance'] < 0, 'Montant_decouvert_dispo'] = \
                dm_bank_account.loc[dm_bank_account['Account_Balance'] < 0, 'Account_Balance'] + \
                dm_bank_account.loc[dm_bank_account['Account_Balance'] < 0, 'Overdraft_Limit_Amt']

            dm_bank_account['Montant_decouvert_non_auto'] = dm_bank_account['Montant_decouvert_dispo'].values
            dm_bank_account.loc[dm_bank_account['Montant_decouvert_non_auto'] > 0, 'Montant_decouvert_non_auto'] = 0

            dm_bank_account.loc[dm_bank_account['Montant_decouvert_dispo'] < 0, 'Montant_decouvert_dispo'] = 0

            df = dm_bank_account[
                ['Date_Valid', 'year', 'month', 'account_id1', 'Account_Balance', 'Overdraft_Limit_Amt',
                 'Contract_Date', 'min_end', 'Montant_decouvert_dispo', 'Montant_decouvert_non_auto']]

            df = df.merge(nb_jour_debit_non_auto[['Host_Agreement_Id_2', 'nb_jour_debit_non_auto']],
                          left_on=['account_id1'], right_on=['Host_Agreement_Id_2'], how='left')

            del dm_bank_account, nb_jour_debit_non_auto

            if file_name + '_{}_{}.csv'.format(p[1], p[0]) in os.listdir(save_path):
                old = pd.read_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]))
                df = pd.concat([old, df])
            df.to_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]), index=False)

            
            
            
def split_files_for_impayes(read_path: str, save_path: str, file_name: str, date_columns: str, chunk_size: int,
                                col_to_convert: str) -> None:
    """
    :param read_path: file to read . This file will be split per month
    :param save_path: folder where to save splitted files
    :param file_name: file name
    :param date_columns: column date according to which to make the split
    :param chunk_size: Number of row in the chunk
    :param col_to_convert: column to convert
    :return: None
    """
    if not os.path.exists(save_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(save_path)
    nb_rows = chunk_size
    
    pret= pd.read_csv("../data/output/pret/all0_pret.csv", sep=';',dtype= {'Contract_Substatus_pret':'string','Contract_Status_pret':'string'})
    pret=pret[['Agreement_Id_pret', 'Party_Id','Contract_Substatus_pret','Contract_Status_pret']]
    pret = pret[pret.Contract_Substatus_pret.isin(['VA','DE'])]
    pret = pret[pret.Contract_Status_pret=='1']
    pret.drop_duplicates(inplace=True)
    
    cpte=pd.read_csv('../data/processed/cpte_cheq_features.csv')#,parse_dates=['New_End_Date']
    cpte=cpte[['Party_Id', 'year', 'month']]#, 'New_End_Date'
    
    
    aux = pd.DataFrame({'year': cpte['year'].values, 'month': cpte['month'].values,
                            'day': [1] * cpte.shape[0]})

    cpte['start_day_of_month'] = pd.to_datetime(aux).values
    del aux
    
    period = cpte.drop_duplicates(subset=['year', 'month'])[['year', 'month','start_day_of_month']]#,'New_End_Date'
    period['start_day_of_month'] = pd.to_datetime(period['start_day_of_month'] )
    #period['New_End_Date'] = pd.to_datetime(period['New_End_Date'] )
    print(period.head())
    ff=0
    for chunk in pd.read_csv(read_path, sep=';', chunksize=nb_rows,
                             na_values=['', ' ', ' ' * 2, ' ' * 3, ' ' * 4, ' ' * 5, '?']):
        print(ff+chunk.shape[0])
        ff+=chunk.shape[0]
        chunk = chunk.loc[chunk['Agreement_Id'].isin(pret['Agreement_Id_pret'].unique())]
        chunk[date_columns] = pd.to_datetime(chunk[date_columns], format='%d.%m.%Y')
        chunk['Start_Date'] = pd.to_datetime(chunk['Start_Date'], format='%d.%m.%Y')
        chunk['Repymt_Date'] = pd.to_datetime(chunk['Repymt_Date'], format='%d.%m.%Y',errors='coerce')
        chunk['End_Date_Rebuild'] = pd.to_datetime(chunk['End_Date'], format='%d.%m.%Y', errors='coerce')
        # this is to deal with the limit of pandas datetime
        chunk['End_Date_Rebuild'] = chunk['End_Date_Rebuild'].fillna(
            datetime.datetime.today().date() + relativedelta(years=40))

        chunk['year'] = chunk[date_columns].dt.year
        chunk['month'] = chunk[date_columns].dt.month
        
        

        chunk['End_Date_Rebuild'] = pd.to_datetime(chunk['End_Date_Rebuild']).values
        chunk=chunk.loc[chunk['Agreement_Id'].isin(pret['Agreement_Id_pret'].unique())]
        for p in period.values:
            # new = chunk.loc[(chunk['year'] == p[0]) & (chunk['month'] == p[1])]
            try:
                new = chunk.loc[((chunk['year'] == p[0]) & (chunk['month'] == p[1])) | (
                    (chunk['End_Date_Rebuild'].dt.year == p[0]) & (chunk['End_Date_Rebuild'].dt.month == p[1])) | (
                                    (chunk['Start_Date'] <= p[2]) & (chunk['End_Date_Rebuild'] >= p[2]))]
            except:
                print(p[0], p[1], p[2])
            
            print(p[0], p[1], p[2])
            new=new.merge(pret,how='left',left_on='Agreement_Id',right_on='Agreement_Id_pret')

            new['year'] = p[0]
            new['month'] = p[1]
            new['New_Start_Date'] = new['Start_Date'].values
            new['New_End_Date'] = new['End_Date'].values
            new.loc[(new['Start_Date'].dt.month != new['month']) | (
                    new['Start_Date'].dt.year != new['year']), 'New_Start_Date'] = datetime.datetime(p[0], p[1], 1)

            new.loc[(new['End_Date_Rebuild'].dt.month != new['month']) | (
                    new['End_Date_Rebuild'].dt.year != new['year']), 'New_End_Date'] = datetime.datetime(p[0], p[1],
                      calendar.monthrange(p[0],p[1])[1])
            print("*********************************2e",p[0], p[1], p[2])
            d_date=pd.to_datetime(p[3])
            past_3month=d_date-relativedelta(months=3)
            past_6month=d_date-relativedelta(months=6)
            past_12month=d_date-relativedelta(months=12)
            
            impayes_3month=new.loc[(new['Repymt_Status']==8)&(new['Start_Date'].dt.date>past_3month.date())]
            impayes_6month=new.loc[(new['Repymt_Status']==8)&(new['Start_Date'].dt.date>past_6month.date())]
            impayes_12month=new.loc[(new['Repymt_Status']==8)&(new['Start_Date'].dt.date>past_12month.date())]
            
            impayes_3month.drop_duplicates(subset=['Agreement_Id','Party_Id','Repymt_Nbr'],inplace=True)
            impayes_3month=impayes_3month.groupby(['Party_Id']).size().reset_index().rename(index=str,columns={0:'nbr_impayes_3month'})
            
            impayes_6month.drop_duplicates(subset=['Agreement_Id','Party_Id','Repymt_Nbr'],inplace=True)
            impayes_6month=impayes_6month.groupby(['Party_Id']).size().reset_index().rename(index=str,columns={0:'nbr_impayes_6month'})
            
            impayes_12month.drop_duplicates(subset=['Agreement_Id','Party_Id','Repymt_Nbr'],inplace=True)
            impayes_12month=impayes_12month.groupby(['Party_Id']).size().reset_index().rename(index=str,columns={0:'nbr_impayes_12month'})
            
            impaye_en_cours= new.groupby(['Agreement_Id','Party_Id','Repymt_Nbr','Repymt_Date'])['Repymt_Status'].max().reset_index()
            impaye_en_cours=impaye_en_cours.loc[(impaye_en_cours['Repymt_Status']==8)]
            impaye_en_cours=impaye_en_cours.groupby(['Party_Id'])['Repymt_Date'].max().reset_index()
            impaye_en_cours['Repymt_Date'] = pd.to_datetime(impaye_en_cours['Repymt_Date'])
            impaye_en_cours['Nbre_jours_retard'] = d_date.date()-impaye_en_cours['Repymt_Date'].dt.date
            impaye_en_cours['Nbre_jours_retard'] = impaye_en_cours['Nbre_jours_retard'].dt.days
            
            del new
            
            new = impayes_3month.merge(impayes_6month,on=['Party_Id'],how='outer')
            new = new.merge(impayes_12month,on=['Party_Id'],how='outer')
            new = new.merge(impaye_en_cours,on=['Party_Id'],how='outer')
            
            try:
                new[col_to_convert] = new[col_to_convert].astype(str).str.strip().str.replace(',', '')
                new[col_to_convert] = new[col_to_convert].astype(int)
            except:
                pass

            if file_name + '_{}_{}.csv'.format(p[1], p[0]) in os.listdir(save_path):
                old = pd.read_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]))
                new = pd.concat([old, new])
            new.to_csv(save_path + '/' + file_name + '_{}_{}.csv'.format(p[1], p[0]), index=False)

