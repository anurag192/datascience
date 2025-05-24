import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformed_object(self):

        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("standardization",StandardScaler())

                ]

            )
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder(handle_unknown='ignore')),
                   
                ]
            )
            logging.info("Pipeline code written")
            preprocessor=ColumnTransformer(
                transformers=[
                     ("numerical_pipeline",numerical_pipeline,numerical_columns),
                     ("categorical_pipeline",categorical_pipeline,categorical_columns)

                ]
               

            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Train and test data obtained")
            preprocessor_obj=self.get_data_transformed_object()

            target_column="math_score"
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)

            target_feature_train_df=train_df[target_column]
            target_feature_test_df=test_df[target_column]

            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
        
    





    