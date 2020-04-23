#!/usr/bin/env python3
from common.preprocess import Preprocess
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sys
import pickle

class StandardScaler():
	def __init__(self,cols):
		self.columns=cols
		self.model=StandardScaler()

	def fit_transform(self,data):
		scaled=self.model.fit_transform(data[self.columns])
		for idx in range(len(self.columns)):
			data[self.columns[idx]]=scaled[:,idx]
		return data

	def fit(self,data):
		self.model.fit(data[self.columns])

	def transform(self,data):
		scaled=self.model.transform(data[self.columns])
		for idx in range(len(self.columns)):
			data[self.columns[idx]]=scaled[:,idx]
		return data


class Standardize(Preprocess):
	def __init__(self):
		super().__init__()

	def standardize(self,data):
		self.model=StandardScaler(cols=self.columns)
		data=self.model.fit_transform(data)
		return data

	def make_parser(self):
		"""
		parse_argsによって内部的に呼ばれる関数です。
		共通オプションを追加するsuper().make_parser()を実行した後、固有オプションを追加したパーサーを返します。
		"""
		# 固有部分
		parser=super().make_parser()
		#parser.add_argument("-c","--columns",dest="columns",help="select colums you wish to normalize",default=None,type=str)
		#parser.add_argument("-a","--all_columns",dest="all_columns",default=False,action="store_const",const=True)
		return parser


	def set_parsed_args_unique(self,parsed):
		"""
		固有のオプションを属性に追加する関数です。

		:param parsed:　コマンドライン引数をパースしたもの
		"""

		#self.columns=parsed.columns
		#self.all_columns=parsed.all_columns

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
		受けたcsv形式のデータフレームに対して、指定された列の標準化を行います
		出力は標準化後のデータフレーム(csv形式)です
		"""
		parsed=self.parse_args(args)

		self.set_parsed_args_common(parsed)
		self.set_parsed_args_unique(parsed)

		#入力ファイル読み込み
		data=self.read_data()
		self.columns=self.get_col_list()

		#標準化
		data.data=self.standardize(data.data)

		#前処理をフローに追加
		data.add_preprocess(self.model)

		"""
		#変換規則のファイル出力
		with open(self.temp_files_path+"standardize.pickle","wb") as f:
			pickle.dump(self.std_dict,f)

		#前処理の順番を保存
		self.write_order()
		"""

		#標準化後のデータセット出力
		self.write_data(data)

if __name__=="__main__":
	std=Standardize()
	std.main(sys.argv[1:])
