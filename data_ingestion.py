import pandas as pd
import os
from logger import logging
import warnings

warnings.filterwarnings("ignore")

class DataIngestion():
    
    def __init__(self):
        
        logging.info(f"Data Ingestion started ..\n")
        
        #Loading the data 
        df_total = pd.read_csv("5g-project-data/ECdata.csv",parse_dates=['Time'])
        df_bs = pd.read_csv("5g-project-data/BSinfo.csv")
        df_cell = pd.read_csv("5g-project-data/CLdata.csv", parse_dates=['Time'])
        
        logging.info(f"Data Ingestion Complete ..\n")
        
        #To confirm whether the data directory exists or not, if not, then it will create one
        os.makedirs('data', exist_ok=True)
        
        #Saving the data as parquet files
        df_total.to_parquet(f'data/total.parquet', engine='pyarrow', index=False)
        df_bs.to_parquet(f'data/bs.parquet', engine='pyarrow', index=False)
        df_cell.to_parquet(f'data/cell.parquet', engine='pyarrow', index=False)
        
        
        

   


