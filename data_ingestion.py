# import necessary libraries
import pandas as pd  
import os  
from logger import logging 
import warnings  

warnings.filterwarnings("ignore")  # ignore all warnings to keep output clean

class DataIngestion():
    
    def __init__(self):
        # log that data ingestion process start
        logging.info(f"Data Ingestion started ..\n")
        
        # Loading the total data from CSV, parse Time column to datetime
        df_total = pd.read_csv("5g-project-data/ECdata.csv", parse_dates=['Time'])
       
        # Loading the base station info data
        df_bs = pd.read_csv("5g-project-data/BSinfo.csv")
       
        # Loading the cell data from CSV, parse Time column
        df_cell = pd.read_csv("5g-project-data/CLdata.csv", parse_dates=['Time'])
        
        # log that data ingestion is complete
        logging.info(f"Data Ingestion Complete ..\n")
        
        # check if 'data' folder exist, if not then create new folder
        os.makedirs('data', exist_ok=True)
        
        # save loaded data into parquet files for fast reading later
        df_total.to_parquet('data/total.parquet', engine='pyarrow', index=False)
        df_bs.to_parquet('data/bs.parquet', engine='pyarrow', index=False)
        df_cell.to_parquet('data/cell.parquet', engine='pyarrow', index=False)