import os
import pickle
import toml
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import get_logger
from sklearn.model_selection import GridSearchCV

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

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model_class in models.items():
           
            model_params = params.get(model_name, {})
            logger.info(f"Model {model_name}, params: {model_params}")
            # model = model_class(**model_params)

            if model_params:
                logger.info(f"HyperParameters Grid Search for: {model_name}..")
                gs = GridSearchCV(estimator=model_class(),
                                  param_grid=model_params,
                                  cv=3,
                                  scoring="r2",
                                  n_jobs=-1)
                gs.fit(X_train, y_train)

                logger.info(f"Best Parameters found: {gs.best_params_}")
                model = gs.best_estimator_
            else:
                logger.info(f"Training {model_name} without hyperparameters..")
                model = model_class()
                model.fit(X_train, y_train)

            logger.info(f"Starting training {model_name}..")
            y_test_preds = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_preds)

            logger.info(f"R2 Score on Test Set: {test_model_score:.2f}")

            report[model_name] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e)

#-----------------------------------------------------------------#
# LOAD CONFIG
#-----------------------------------------------------------------#

def load_config(path: str):
    return toml.load(path)
