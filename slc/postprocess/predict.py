#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import pickle
import sys
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../modeling"))
from classification import *
from regression import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from dataset import Dataset
import dill

class Predict():
    def __init__(self):
        pass
    def make_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", dest="input",
                            help="setting input file path", metavar="FILE",type=str)
        parser.add_argument("-o", "--output", dest="output",
                            help="setting output file path", metavar="FILE", default=sys.stdout)
        parser.add_argument("-m", "--model", dest="model_path",
                            metavar="FILE", type=str, help="set model path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../temp_files/model.pickle"))
        parser.add_argument("-p","--probability",dest="probability",help="set probability on",action="store_const",const=True,default=False)
        parser.add_argument("-t","--include_target_column",dest="include_target_column",help="use this option if your input dataset include target column",action="store_const",const=True,default=False)

        return parser#.parse_args(args)

    def parse_args(self,args):
        parser=self.make_parser()

        return parser.parse_args(args)

    def read_model(self,model_path):
        """
        標準入力もしくはコマンドライン引数からモデルを読み込む
        """
        # inputがstdinかファイル指定かで場合分け
        with open(model_path, "rb") as f:
            return dill.load(f)

    def set_output(self,merged,parsed_output):
        """
        マージしたデータセットを.csvまたは標準出力に出力する
        """
        # outputがstdoutかファイル出力かで場合分け
        if type(parsed_output) == str:
            #output = open(parsered.output, 'wb')
            merged.to_csv(parsed_output,index=False)  # 出力先が指定されている場合、csv形式で出力する
        else:
            output = parsed_output.buffer  # されていない場合は標準出力
            output.write(pickle.dumps(merged))
        return

    def preprocessing(self,x_test):
        """
        前処理で作成した変換規則から、テストデータのクラスラベルをベクトルに変換する
        preprocessingをここに並べていく(要検討)

        """
        for idx in range(len(self.model.preprocess)):
            x_test=self.model.preprocess[idx].transform(x_test)
        return x_test


    def main(self,args):
        parsed = self.parse_args(args)

        # testデータ読み込み
        test_data_original = Dataset(pd.read_csv(parsed.input))
        #モデル読み込み
        self.model=self.read_model(parsed.model_path)
        #予測を確率で出すかどうかの設定(classificationクラスのみ有効)
        self.model.probability=parsed.probability
        #一時ファイルのpath
        #self.temp_files_path=parsed.temp_dir
        #クラスラベルの処理
        test_data_preprocessed=Dataset(self.preprocessing(test_data_original.data))
        #ターゲット列の分離
        self.include_target_column=parsed.include_target_column
        if self.include_target_column:
            target_col_name=self.model.target_colname
            y_test=test_data_preprocessed.data[target_col_name]
            test_data_preprocessed.data=test_data_preprocessed.data.drop(columns = target_col_name)
        #予測
        pred_df=self.model.predict(test_data_original.data,test_data_preprocessed.data)
        if self.include_target_column:
            pred_df=pd.concat([pred_df,y_test],axis=1)
        merged=pd.concat([pred_df,test_data_original.data],axis=1)
        #出力
        self.set_output(merged,parsed.output)

        return

if __name__=="__main__":
    pred=Predict()
    pred.main(sys.argv[1:])
    #main(sys.argv[1:])
