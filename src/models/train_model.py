import os

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")


def train_model(df_path):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("spaceship-titanic")

    with mlflow.start_run():
        mlflow.get_artifact_uri()

        df = pd.read_csv(df_path)

        df['HomePlanet'] = df['HomePlanet'].astype('category')
        df['Destination'] = df['Destination'].astype('category')
        df['Deck'] = df['Deck'].astype('category')
        df['Side'] = df['Side'].astype('category')

        X_train, X_val, y_train, y_val = train_test_split(
            df.drop('Transported', axis=1),
            df['Transported'],
            test_size=0.2, random_state=42
        )
        cat_features = np.where(X_train.dtypes == 'category')[0]

        best_params = {'iterations': 47,
                       'learning_rate': 0.027583475549166746,
                       'depth': 4,
                       'l2_leaf_reg': 1.0551779964424746e-05,
                       'bootstrap_type': 'Bayesian',
                       'random_strength': 2.0931628460945333e-07,
                       'bagging_temperature': 0.923385947687978,
                       'od_type': 'Iter',
                       'od_wait': 26}

        mlflow.log_params(best_params)

        clf_model = CatBoostClassifier(iterations=best_params.get('iterations'),
                                       learning_rate=best_params.get('learning_rate'),
                                       depth=best_params.get('depth'),
                                       l2_leaf_reg=best_params.get('l2_leaf_reg'),
                                       bootstrap_type=best_params.get('bootstrap_type'),
                                       random_strength=best_params.get('random_strength'),
                                       bagging_temperature=best_params.get('bagging_temperature'),
                                       od_type=best_params.get('od_type'),
                                       od_wait=best_params.get('od_wait'))

        clf_model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))

        mlflow.catboost.log_model(clf_model, "model")

        y_pred_val = clf_model.predict(X_val)
        accuracy = np.mean(y_pred_val == y_val)
        mlflow.log_metric("accuracy", accuracy)

    return clf_model


if __name__ == "__main__":
    clf = train_model('../../data/processed/train_processed.csv')
