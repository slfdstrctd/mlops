import pandas as pd

from train_model import train_model


def predict_model(clf_model, test_path, submission_path):
    test = pd.read_csv(test_path)
    ids = test['PassengerId']

    cat_features = ['HomePlanet', 'Destination', 'Deck', 'Side']

    test[cat_features] = test[cat_features].apply(lambda x: x.astype('category'))

    preds = clf_model.predict(test.drop(['PassengerId'], axis=1))  # 'Transported'

    res_df = pd.concat([ids, pd.Series(preds)], axis=1)
    res_df.columns = ['PassengerId', 'Transported']
    res_df.to_csv(submission_path, index=False)

    return res_df


if __name__ == "__main__":
    clf = train_model('../../data/processed/train_processed.csv')
    predict_model(clf, '../../data/processed/test_processed.csv', 'submission.csv')
