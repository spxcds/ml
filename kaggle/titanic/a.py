import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def data_process(df):
    dataframe = df
    dataframe.Age = df.Age = df.Age.fillna(-999)
    dataframe.Sex = df.Sex.map({'male': 0, 'female': 1})
    dataframe.Embarked = df.Embarked.map({'S': 0, 'C': 1, 'Q': 2, None: 3})

    cabin_map = {}
    cabin = df.Cabin.fillna('Z0')
    cabin_number = [i[0] for i in cabin]
    for i, k in enumerate(set(cabin_number)):
        cabin_map[k] = i

    dataframe.Cabin = cabin_number
    dataframe.Cabin = dataframe.Cabin.map(cabin_map)

    return dataframe.drop(columns=['PassengerId', 'Name', 'Ticket'])


def main():
    dataset = pd.read_csv('train.csv')
    dataset = data_process(df=dataset)
    train = dataset.set_index('Survived')
    print train.shape
    print train.head()
    print train.columns

    # train, test = train_test_split(dataset, test_size=0.3)

    # lr = LogisticRegression(penalty='l2', C=1.0, verbose=True)
    lr = LogisticRegression(penalty='l2', C=1.0)
    lr.fit(train, dataset["Survived"])

    test = pd.read_csv('test.csv')
    test = data_process(df=test)
    test = test.fillna(-999)
    print test.shape
    print test.head()
    print test.columns

    predict = lr.predict(test)

    for x in list(predict):
        print x


if __name__ == '__main__':
    main()
