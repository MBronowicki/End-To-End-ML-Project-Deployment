import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModeTrainer

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
        
if __name__=="__main__":

    obj=DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(train_path)

    data_transformation=DataTransformation()
    train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_path, test_path)

    logger.info(f"train_arr: {train_arr.shape}")
    logger.info(f"test_arr: {test_arr.shape}")

    model_trainer = ModeTrainer()

    best_model_name, best_model_r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
