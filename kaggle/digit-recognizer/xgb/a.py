import xgboost as xgb
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.externals import joblib
import xgboost as xgb


def load_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, model_name=None):
    if not model_name:
        model_name = 'model-' + str(int(time.time()))
    joblib.dump(model, model_name)


def evaluate(model, file_path='./test.csv'):
    test = load_data(file_path=file_path)
    y_true = test['label']

    y_predict = model.predict(test.set_index('label'))
    print accuracy_score(y_true=y_true, y_pred=y_predict)


def train(file_path='./train.csv'):
    train_data_path = file_path
    df = load_data(file_path=train_data_path)
    train = df.set_index('label')

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.3, verbose=True)

    gb.fit(train, df['label'])
    return gb


def predict(model, file_name='result.csv'):
    test = load_data(file_path='./predict.csv')
    predict = model.predict(test)
    with open(file_name, 'w') as f:
        f.write('ImageId,Label\n')
        k = 1
        for v in list(predict):
            result = str(k) + ',' + str(v) + '\n'
            k += 1
            f.write(result)


def main():
    model = xgb.Booster()

    model.load_model(fname='./trained.model')

    print model

    test = pd.read_csv('test.csv', header=None)
    print test.shape

    pred = model.predict(xgb.DMatrix(test, missing=0))

    i = 1
    for v in list(pred):
        print str(i) + ',' + str(int(v))
        i += 1


if __name__ == '__main__':
    main()
