import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import numpy as np
from src.features.build_features import preprocess


def train_model(df_path):
    df = pd.read_csv(df_path)
    df_train = preprocess(df)

    X_train, X_val, y_train, y_val = train_test_split(df_train.drop('Transported', axis=1), df_train['Transported'],
                                                      test_size=0.2, random_state=42)
    cat_features = np.where(X_train.dtypes == 'category')[0]

    best_params = {'iterations': 475,
                   'learning_rate': 0.027583475549166746,
                   'depth': 4,
                   'l2_leaf_reg': 1.0551779964424746e-05,
                   'bootstrap_type': 'Bayesian',
                   'random_strength': 2.0931628460945333e-07,
                   'bagging_temperature': 0.923385947687978,
                   'od_type': 'Iter',
                   'od_wait': 26}

    clf = CatBoostClassifier(iterations=best_params.get('iterations'),
                             learning_rate=best_params.get('learning_rate'),
                             depth=best_params.get('depth'),
                             l2_leaf_reg=best_params.get('l2_leaf_reg'),
                             bootstrap_type=best_params.get('bootstrap_type'),
                             random_strength=best_params.get('random_strength'),
                             bagging_temperature=best_params.get('bagging_temperature'),
                             od_type=best_params.get('od_type'),
                             od_wait=best_params.get('od_wait'))

    clf.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))

    return clf


if __name__ == "__main__":
    clf = train_model('/data/raw/train.csv')
