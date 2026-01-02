import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger()

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Initialize data ingestion...")
        try:
            logger.info("Read the dataset as dataframe")
            df=pd.read_csv("./data/stud.csv")

            logger.info("Split Raw Data into Train and Test set")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info("Ingestion Data Competed! Successfully!!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,

            )
        except Exception as e:
            raise CustomException(e)

