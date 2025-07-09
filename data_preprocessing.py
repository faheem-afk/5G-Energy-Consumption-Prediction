from utils import \
                load_data, \
                seed_everything,\
                periodic_spline_transformer, \
                add_sosfiltfilt,\
                add_sg,\
                set_seed, \
                read_from_yaml

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from logger import logging
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessing():
    
    def __init__(self):
        
        #Creating necessary object attributes
        self.df_total, self.df_bs, self.df_cell = load_data()
        self.num_ru_categories: int
        self.num_mo_categories: int

    def pre_processing(self):
        
        #Setting seed value for reproducibility
        set_seed(42)
        seed_everything(seed=42)
        
        #To confirm whether the data directory exists or not, if not, then it will create one
        os.makedirs('data/preprocessed_data', exist_ok=True)

        #To confirm if the file exists or not, if it already does, then no need to preprocess again
        if os.path.exists('data/preprocessed_data/train.parquet'): 
            
            train_data = pd.read_parquet('data/preprocessed_data/train.parquet', engine='pyarrow')
            test_data = pd.read_parquet('data/preprocessed_data/test.parquet', engine='pyarrow')
            
            self.num_ru_categories = train_data['RUType'].unique()
            self.num_mo_categories = train_data['Mode'].unique()

            return train_data, test_data
        
        #If the file doesn't already exist, then we start preprocessing the data
        logging.info(f"Data Preprocessing Started ..\n")
        
        #Merging the base_station and CellName dataframes 
        df_features = self.df_cell.merge(self.df_bs, on=['BS', 'CellName'], how = 'outer')
        
        #Removing the imbalance in the data by only using the majority cellName, i.e.., Cell0
        df_features = df_features[df_features['CellName'] == 'Cell0']
        df_features.reset_index(drop=True, inplace=True)
        
        #Merging the features and df_total dataframes to create a unit 
        self.df_total = self.df_total.merge(df_features, on=['BS', 'Time'], how='left')
        
        #Using regular expressions to clean the data
        self.df_total['BS'] = self.df_total['BS'].str.replace(r'[a-zA-Z_]', '', regex=True).astype(int)
        for col in ['RUType','Mode']:
            self.df_total[col] = self.df_total[col].str.replace(r'[a-zA-Z]', '', regex=True).astype(int)

        #Sorting the data using base_station and time columns
        self.df_total.sort_values(['BS','Time'], ascending=True,ignore_index=True,inplace=True)
        
        #Extracting features like day, weekday_number etc from the time column
        self.df_total['day'] = self.df_total['Time'].dt.day
        self.df_total['weekday_number'] = self.df_total['Time'].dt.weekday
        self.df_total['hour'] = self.df_total['Time'].dt.hour
        
        hour_df = self.df_total[['hour']].copy()
        
        #Creating splines using the time column to show periodicity
        splines = periodic_spline_transformer(24, n_splines=12).fit_transform(hour_df)
        splines_df = pd.DataFrame(splines,columns=[f"hour_spline_{i}" for i in range(splines.shape[1])])
        self.df_total = pd.concat([self.df_total,splines_df],axis=1)

        self.df_total = self.df_total.sort_values(['BS','Time'],ascending=True,ignore_index=True)
        
        #Creating lags in the data using the features that frequently vary with time
        all_shits = list(np.arange(1,4)) # 
        for shift_i in tqdm(all_shits):
            for col in ['load','ESMode1','ESMode2','ESMode3','ESMode6','Time','Energy']:
                self.df_total[f'{col}_T-{shift_i}'] = self.df_total.groupby(['BS'])[col].shift(shift_i)        
        
        for shift_i in tqdm(all_shits):
            self.df_total[f'Time_T-{shift_i}_hours_elapsed'] = (self.df_total[f'Time_T-{shift_i}']-self.df_total['Time']).dt.total_seconds() / 3600
            del self.df_total[f'Time_T-{shift_i}']
        
        #Creating a load bin column to show the actual trend of load column with energy
        num_bins = 100
        self.df_total['load_bin'] = pd.cut(self.df_total['load'],bins=[round(i,2) for i in list(np.arange(0,1.01,0.01))],labels=[f'{i}' for i in range(num_bins)])
        self.df_total['load_bin'] = self.df_total['load_bin'].astype(float)
        
        #Imputing the missing value with 0
        self.df_total['load_bin'] = self.df_total['load_bin'].fillna(0)
        
        #Using Savitz-Golay filter and the butterworth filter to create smoothened load features
        add_sg(self.df_total)
        add_sosfiltfilt(self.df_total)
        
        #Using Feature Engineering to create aggragated features from feature lags
        self.df_total['Energy_lagged_mean'] = self.df_total[['Energy_T-3','Energy_T-2','Energy_T-1']].mean(axis=1)
        self.df_total['Energy_lagged_std'] = self.df_total[['Energy_T-3','Energy_T-2','Energy_T-1']].std(axis=1)

        self.df_total['load_lagged_mean'] = self.df_total[['load_T-3','load_T-2','load_T-1']].mean(axis=1)
        self.df_total['load_lagged_std'] = self.df_total[['load_T-3','load_T-2','load_T-1']].std(axis=1)
        
        #Using linear combination of features to build more relevant ones
        self.df_total['load*Mode'] = self.df_total['load'] * self.df_total['Mode']
        self.df_total['Antenna*tx'] = self.df_total['Antennas'] * self.df_total['TXpower']
        self.df_total['Bandwidth*freq'] = self.df_total['Bandwidth'] * self.df_total['Frequency']


        self.df_total['Energy_lagged_sum'] = self.df_total[['Energy_T-3','Energy_T-2','Energy_T-1']].sum(axis=1)
        self.df_total['Energy_lagged_max'] = self.df_total[['Energy_T-3','Energy_T-2','Energy_T-1']].max(axis=1)
        self.df_total['Energy_lagged_min'] = self.df_total[['Energy_T-3','Energy_T-2','Energy_T-1']].min(axis=1)

        self.df_total['load_lagged_sum'] = self.df_total[['load_T-2','load_T-1']].sum(axis=1)
        self.df_total['load_lagged_max'] = self.df_total[['load_T-3','load_T-2','load_T-1']].max(axis=1)
        self.df_total['load_lagged_min'] = self.df_total[['load_T-3','load_T-2','load_T-1']].min(axis=1)
        
        lagged_features, static_data, _ = read_from_yaml('variables.yaml')

        cat_cols = ['RUType', 'Mode'] 

        target_col = 'Energy'

        needed = static_data + lagged_features + cat_cols + [target_col]

        df = self.df_total.dropna(subset=needed).copy()

        df = df.reset_index(drop=True)
        
        #Encoding the categorical columns
        le_ru = LabelEncoder()
        le_mo = LabelEncoder()
        
        df['RUType_enc'] = le_ru.fit_transform(df['RUType'].values)
        df['Mode_enc'] = le_mo.fit_transform(df['Mode'].values)

        self.num_ru_categories = len(le_ru.classes_)
        self.num_mo_categories = len(le_mo.classes_)

        numeric_to_scale = static_data + lagged_features
        
        #Scaling the numerical features in the data for a stable gradient 
        scaler = StandardScaler()
        df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale].values)
        
        #Splitting the data into train test 
        train_data, test_data = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
        train_data.reset_index(inplace=True, drop=True)

        test_data.reset_index(inplace=True, drop=True)

        #Saving the data as preprocessed parquet files
        train_data.to_parquet(f'data/preprocessed_data/train.parquet', engine='pyarrow', index=False)
        test_data.to_parquet(f'data/preprocessed_data/test.parquet', engine='pyarrow', index=False)
        
        print("\n")
        
        logging.info(f"Data Preprocessing Complete ..\n")
        
        return train_data, test_data
      




        













