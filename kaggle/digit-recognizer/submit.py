import xgboost as xgb
import os
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.externals import joblib
import sys

sys.path.append('../lib')

from kaggle_api import KaggleAPI


def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'label' in df:
        print('label in df')
        df[df.iloc[:, 1:] > 0] = 1
    else:
        print('label not in df')
        df[df > 0] = 1
    return df


def save_model(model, model_name=None):
    if not model_name:
        model_name = 'model/model-' + str(int(time.time()))
    joblib.dump(model, model_name)


def load_model(model_path):
    return joblib.load(model_path)


def evaluate(model, file_path='./test.csv'):
    test = load_data(file_path=file_path)
    y_true = test['label']

    y_predict = model.predict(test.set_index('label'))
    print(accuracy_score(y_true=y_true, y_pred=y_predict))


def train(file_path='./train.csv',
          n_estimators=10,
          max_depth=10,
          learning_rate=0.1):
    train_data_path = file_path
    df = load_data(file_path=train_data_path)
    train = df.set_index('label')

    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        # n_jobs=-1,
        verbose=True)

    gb.fit(train, df['label'])
    return gb


def predict(model, file_name='result.csv'):
    test = load_data(file_path='./predict.csv')
    # print('--------' * 20)
    # print(test)
    # print(test.shape)
    # print(list(test.loc[0]))
    predict = model.predict(test)
    with open(file_name, 'w') as f:
        f.write('ImageId,Label\n')
        k = 1
        for v in list(predict):
            result = str(k) + ',' + str(v) + '\n'
            k += 1
            f.write(result)


def get_result(model_name, result_file_name):
    model = load_model(model_name)
    predict(model, file_name=result_file_name)


def traverseDirByOSWalk(path):
    # path = os.path.expanduser(path)

    file_list = []
    for (dirname, subdir, subfile) in os.walk(path):
        print('[' + dirname + ']')
        for f in subfile:
            # print(os.path.join(dirname, f))
            file_list.append(os.path.join(dirname, f))
    return file_list


if __name__ == '__main__':
    path = 'trained_model'
    file_list = traverseDirByOSWalk(path=path)

    for file_name in file_list:
        predict_file_name = 'predict_result/' + file_name.split(
            '/')[-1] + '.txt'
        # print file_name, predict_file_name
        get_result(model_name=file_name, result_file_name=predict_file_name)
        break

        kaggle_api = KaggleAPI()
        kaggle_api.submit(
            competition='digit-recognizer',
            file_name=predict_file_name,
            message=file_name.split('/')[-1])
