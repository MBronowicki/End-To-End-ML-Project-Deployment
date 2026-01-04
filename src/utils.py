import os
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import get_logger

logger = get_logger()

#-----------------------------------------------------------------#
# SAVE OBJECT
#-----------------------------------------------------------------#

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e)
    
#-----------------------------------------------------------------#
# MODEL EVALUATION
#-----------------------------------------------------------------#

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            logger.info(f"Starting training {model_name}..")
            model.fit(X_train, y_train)

            y_test_preds = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_preds)

            logger.info(f"R2 Score on Test Set: {test_model_score:.2f}")

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e)

