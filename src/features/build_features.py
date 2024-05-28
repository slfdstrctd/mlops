import pandas as pd

def preprocess(df):
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)

    df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num']] = df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num']].fillna(value=0)

    df['CryoSleep'] = df['CryoSleep'].astype('int')
    df['VIP'] = df['VIP'].astype('int')
    df['Cabin_num'] = df['Cabin_num'].astype('int')

    df[['HomePlanet', 'Destination', 'Deck', 'Side']] = df[['HomePlanet', 'Destination', 'Deck', 'Side']].fillna(value='')

    df['HomePlanet'] = df['HomePlanet'].astype('category')
    df['Destination'] = df['Destination'].astype('category')
    df['Deck'] = df['Deck'].astype('category')
    df['Side'] = df['Side'].astype('category')

    df = df.drop(['Cabin', 'PassengerId', 'Name'], axis=1)

    return df
