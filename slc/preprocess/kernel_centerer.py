#!/usr/bin/env python3
from common.preprocess import Preprocess
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
import json
import sys
import argparse
import numpy as np
import pandas as pd
import pickle

class kc():
	def __init__(self,cols,metric):
		self.columns=cols
		self.metric=metric
		self.model=KernelCenterer()

	def fit(self,data):
		k=pairwise_kernels(data[self.columns],metric=self.metric)
		self.model.fit(k)
	def fit_transform(self,data):
		k=pairwise_kernels(data[self.columns],metric=self.metric)
		transformed=self.model.fit_transform(k)
		for idx in range(len(self.columns)):
			data[self.columns[idx]]=transformed[:,idx]
		return data
	def transform(self,data):
		k=pairwise_kernels(data[self.columns],metric=self.metric)
		transformed=self.model.transform(k)
		for idx in range(len(self.columns)):
			data[self.columns[idx]]=transformed[:,idx]
		return data

class Kernel_centerer(Preprocess):
	def __init__(self):
		super().__init__()

	def kernel_centerer(self,data):
		"""
		複数のカテゴリ変数をベクトル化して、それぞれ変換規則を保存する関数です。
		ベクトル化したデータセットを返します。
		変換規則はenc_dictに保存されています。

		:param data: 学習で用いるデータセット(Dataset型の属性dataを受け取る)
		"""
		self.model=kc(cols=self.columns,metric=self.metric)
		transformed=self.model.fit_transform(data)
		return transformed


	def make_parser(self):
		"""
		parse_argsによって内部的に呼ばれる関数です。
		共通オプションを追加するsuper().make_parser()を実行した後、固有オプションを追加したパーサーを返します。
		"""
		# 固有部分
		parser=super().make_parser()

		parser.add_argument("-m","--metric",dest="metric",default="linear",choices=["rbf", "sigmoid", "polynomial", "poly", "linear", "cosine"])
		return parser


	def set_parsed_args_unique(self,parsed):
		"""
		固有のオプションを属性に追加する関数です。

		:param parsed:　コマンドライン引数をパースしたもの
		"""

		self.metric=parsed.metric

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
		data.data=self.kernel_centerer(data.data)

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
	kc=Kernel_centerer()
	kc.main(sys.argv[1:])
