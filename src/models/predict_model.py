import pandas as pd

from train_model import train_model
from src.features.build_features import preprocess


def predict_model(clf_model, test_path, submission_path):
    test = pd.read_csv(test_path)
    ids = test['PassengerId']
    X_test = preprocess(test)

    preds = clf_model.predict(X_test)
    res_df = pd.concat([ids, pd.Series(preds)], axis=1)
    res_df.columns = ['PassengerId', 'Transported']
    res_df.to_csv(submission_path, index=False)

    return res_df


if __name__ == "__main__":
    clf = train_model('../../data/raw/train.csv')
    predict_model(clf, '../../data/raw/test.csv', 'submission.csv')
