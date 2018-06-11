import xgboost as xgb
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.externals import joblib


def load_data(file_path):
    return pd.read_csv(file_path)


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
    predict = model.predict(test)
    with open(file_name, 'w') as f:
        f.write('ImageId,Label\n')
        k = 1
        for v in list(predict):
            result = str(k) + ',' + str(v) + '\n'
            k += 1
            f.write(result)


# def main():
#     print 'Training...'
#     model = train(file_path='my_test.csv')
#     print 'Training finished!'
#
#     print 'Saving...'
#     save_model(model=model)
#
#     print 'Evaluating...'
#     evaluate(model, file_path='my_test.csv')
#     print 'Done!'


def get_result(model_name):
    model = load_model(model_name)
    predict(model)


if __name__ == '__main__':
    # hyper_param_file = 'hyper_param.txt'
    # with open(hyper_param_file, 'r') as f:
    #     for line in f.readlines():
    #         param_list = line.strip().split()
    #         n_estimators = int(param_list[0])
    #         max_depth = int(param_list[1])
    #         learning_rate = float(param_list[2])
    #         model_name = 'model/model_' + str(n_estimators) + '_' + str(
    #             max_depth) + '_' + str(learning_rate) + '_' + str(
    #                 int(time.time()))

    #         print(model_name)
    #         model = train(
    #             file_path='my_train.csv',
    #             n_estimators=n_estimators,
    #             max_depth=max_depth,
    #             learning_rate=learning_rate)
    #         save_model(model=model, model_name=model_name)
    #         evaluate(model, file_path='my_test.csv')
    get_result(model_name='model/model_140_7_0.1_1528690818')
