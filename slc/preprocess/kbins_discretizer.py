#!/usr/bin/env python3
from common.preprocess import Preprocess
from sklearn.preprocessing import KBinsDiscretizer
import json
import sys
import argparse
import numpy as np
import pandas as pd
import pickle

class kbd():
	def __init__(self,cols,n_bins,encode,strategy):
		self.columns=cols
		self.model=KBinsDiscretizer(n_bins=n_bins,encode=encode,strategy=strategy)

	def fit(self,data):
		self.model.fit(data[self.columns])
	def fit_transform(self,data):
		transformed=self.model.fit_transform(data[self.columns])
		for idx in range(len(self.columns)):
			data[self.columns[idx]]=transformed[:,idx]
		return data
	def transform(self,data):
		transformed=self.model.transform(data[self.columns])
		for idx in range(len(self.columns)):
			data[self.columns[idx]]=transformed[:,idx]
		return data

class Kbins_discretizer(Preprocess):
	def __init__(self):
		super().__init__()

	def kbd(self,data):
		"""
		複数のカテゴリ変数をベクトル化して、それぞれ変換規則を保存する関数です。
		ベクトル化したデータセットを返します。
		変換規則はenc_dictに保存されています。

		:param data: 学習で用いるデータセット(Dataset型の属性dataを受け取る)
		"""
		self.model=kbd(cols=self.columns,n_bins=self.n_bins,encode=self.encode,strategy=self.strategy)
		transformed=self.model.fit_transform(data)
		return transformed


	def make_parser(self):
		"""
		parse_argsによって内部的に呼ばれる関数です。
		共通オプションを追加するsuper().make_parser()を実行した後、固有オプションを追加したパーサーを返します。
		"""
		# 固有部分
		parser=super().make_parser()
		parser.add_argument("-n","--n_bins",dest="n_bins",default=5,type=int)
		parser.add_argument("--encode",dest="encode",default="onehot",choices=["onehot", "onehot-dense", "ordinal"])
		parser.add_argument("-s","--strategy",dest="strategy",default="quantile",choices=["uniform", "quantile", "kmeans"])
		return parser


	def set_parsed_args_unique(self,parsed):
		"""
		固有のオプションを属性に追加する関数です。

		:param parsed:　コマンドライン引数をパースしたもの
		"""

		self.n_bins=parsed.n_bins
		self.encode=parsed.encode
		self.strategy=parsed.strategy

	def parse_args(self,args):
		"""
		コマンドライン引数をパースする関数です。
		すべてのオプションをパースしたものを返します。

		:param args: コマンドライン引数
		"""
		parser=self.make_parser()
		return parser.parse_args(args)

	def main(self,args):
		"""
		メイン関数です
		受けたcsv形式のデータフレームに対して、指定された列のラベルエンコードを行います
		出力はラベルエンコード後のデータフレーム(csv形式)です
		また、変換規則に関するファイルlabel.jsonが生成されます
		上記ファイルをpredict.py実行時に指定してください
		"""

		parsed=self.parse_args(args)

		self.set_parsed_args_common(parsed)
		self.set_parsed_args_unique(parsed)

		#入力ファイル読み込み
		data=self.read_data()
		self.columns=self.get_col_list()

		#ラベルエンコード
		data.data=self.kbd(data.data)

		#前処理をフローに追加
		data.add_preprocess(self.model)

		"""
		#変換規則のファイル出力
		with open(self.temp_files_path+"label.pickle","wb") as f:
			pickle.dump(self.enc_dict,f)

		#前処理の順番を保存
		self.write_order()
		"""

		#エンコード後のデータセットの出力
		self.write_data(data)

if __name__=="__main__":
	kbd=Kbins_discretizer()
	kbd.main(sys.argv[1:])
