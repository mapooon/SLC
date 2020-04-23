import argparse
import numpy as np
import pandas as pd
from sklearn import metrics,svm
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
import sklearn
import pickle
import sys
from jinja2 import Template


# scoring_typeを定義
s_type = [
    "accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_samples",
    "f1_weighted",
    "recall",
    "recall_macro",
    "recall_micro",
    "recall_samples",
    "recall_weighted",
    "neg_log_loss",
    "precision",
    "precision_macro",
    "precision_micro",
    "precision_samples",
    "precision_weighted"
]


class range_check(object):  # 引数が適切かチェックする用

    def __init__(self, low_limit=None, high_limit=None, vtype="integer"):
        self.min = low_limit
        self.max = high_limit
        self.type = vtype

    def __contains__(self, val):
        ret = True
        if self.min is not None:
            ret = ret and (val >= self.min)
        if self.max is not None:
            ret = ret and (val <= self.max)
        return ret

    def __iter__(self):
        low = self.min
        if low is None:
            low = "-inf"
        high = self.max
        if high is None:
            high = "+inf"
        L1 = self.type
        L2 = " {} <= n <= {}".format(low, high)
        return iter((L1, L2))


def SLCsvm(train, target_colname, scoring, n_splits, kfold_type, C, kernel, gamma):
    '''
    n_split>1のときはcvを使ってスコアをcsvで出力する
    戻り値はすべての訓練データを使って学習したモデル
    '''
    # モデルの生成
    clf = sklearn.svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    # target_colname列以外を特徴量とする
    x_train = train.drop(target_colname, axis=1)
    y_train = train.loc[:, target_colname]

    # cv
    if n_splits > 1:
        scores = cross_validate(
            clf, x_train, y_train, scoring=scoring, cv=n_splits, return_train_score=True)
        # scoreをcsvで出力
        pd.DataFrame(scores)[["train_score", "test_score"]
                             ].to_csv("scores.csv")

    # モデルの学習
    model = clf.fit(X=x_train, y=y_train)
    return model


def parseargs(args):  # 引数設定
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help="setting input file", metavar="FILE", default=sys.stdin, type=open)
    parser.add_argument("-o", "--output", dest="output",
                        help="setting output file", metavar="FILE", default=sys.stdout)
    parser.add_argument("-n", "--n_splits", dest="n_splits",
                        default=0, type=int, help="setting n_splits in cv", choices=range_check(low_limit=0))
    parser.add_argument("-t", "--target_colname", dest="target_colname",
                        default="label", type=str, help="setting target_colname")
    parser.add_argument(
        "-s", "--scoring", dest="scoring", default="accuracy", type=str, help="setting scoring type",
        choices=s_type
    )
    parser.add_argument("--kfold_type", dest="kfold_type", default="normal",
                        help="setting kfold type", choices=["normal", "stratified"])
    # 固有部分のparse

    parser.add_argument("-c", "--C", dest="C", help="setting penalty parameter C of the error term", default=1.0, type=float)

    parser.add_argument("-k", "--kernel", dest="kernel", default="rbf", type=str, help="setting kernel function", choices=["linear", "poly", "rbf", "sigmoid", "precomputed"])

    parser.add_argument("-g", "--gamma", dest="gamma", default=-1, type=float, help="setting kernel coefficient")


    return parser.parse_args(args)


def main(args):
    # settings
    # 共通部分
    parsered = parseargs(args)
    n_splits = parsered.n_splits
    target_colname = parsered.target_colname
    scoring = parsered.scoring
    kfold_type = parsered.kfold_type
    # 固有部分

    kernel=parsered.kernel

    C=parsered.C

    gamma="auto" if parsered.gamma == -1 else parsered.gamma


# inputがstdinかファイル指定かで場合分け
    if parsered.input.name != '<stdin>':
        train_data = pd.read_csv(parsered.input)
    else:
        train_data = pickle.loads(
            parsered.input.buffer.read())  # 今のところこの仕様とする

    # ファイル書き込みかstdoutかで場合分け

    if type(parsered.output) == str:
        output = open(parsered.output, 'wb')
    else:
        output = parsered.output.buffer

    # 出力
    model = SLCsvm(train_data, target_colname,
                         scoring, n_splits, kfold_type, C, kernel, gamma)
    output.write(pickle.dumps(model))

    if type(parsered.output) == str:
        output.close()


if __name__ == '__main__':
    main(sys.argv[1:])