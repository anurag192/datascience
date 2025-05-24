import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models
from src.utils import save_object
from sklearn.model_selection import train_test_split

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("train-test split initiated")

            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]

            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "KNeighbors":KNeighborsRegressor(),
                "Adaboost":AdaBoostRegressor()
            }
            param_grid = {
                    "Random Forest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "bootstrap": [True, False]
                    },
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                    "Gradient Boosting": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2]
                    },
                    "Linear Regression": {
                        # Usually no hyperparameters for LinearRegression, unless using regularized variants
                        "fit_intercept": [True, False],
                        "positive": [True, False]
                    },
                    "KNeighbors": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                        "leaf_size": [20, 30, 40]
                    },
                    "Adaboost": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1],
                        "loss": ["linear", "square", "exponential"]
                    }
             }
            

            model_report:dict=evaluate_models(models,X_train,y_train,X_test,y_test,param_grid)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<=0.6:
                raise CustomException("No best model")
            
            logging.info(f"Best model found as {best_model_name} with {best_model_score}")
            save_object(self.model_trainer_config.trained_model_file_path,best_model)

            predict=best_model.predict(X_test)
            score=r2_score(y_test,predict)
            return score


           

        except Exception as e:
            raise CustomException(e,sys)



