import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object, evaluate_models

logger = get_logger()

@dataclass
class ModelTraningConfig:
    train_model_path=os.path.join("artifacts", "best_model.pkl")


class ModeTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTraningConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting train and test input data.")
            X_train,y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regression": KNeighborsRegressor(),
                "XGB Regression": XGBRegressor(),
                # "CatBoosting Regression": CatBoostRegressor(),
                "AdaBoost Regression": AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,
                                              X_test=X_test, y_test=y_test,
                                              models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logger.info(f"Best found model on testing dataset")
            logger.info(f"Best Model Name: {best_model_name}")
            logger.info(f"The best model R2 Score : {best_model_score:.2f}")

            save_object(
                file_path=self.model_trainer_config.train_model_path,
                obj=best_model
            )

            # predicted = best_model.predict(X_test)
            # r2 = r2_score(y_test, predicted)
            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e)