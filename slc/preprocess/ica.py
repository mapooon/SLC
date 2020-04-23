#!/usr/bin/env python3
from common.preprocess import Preprocess
from sklearn.decomposition import FastICA
import numpy as np
import pandas as pd
import sys
import pickle

class ica():
	def __init__(self,cols,n_components):
		self.n_components=n_components
		self.model=FastICA(n_components=self.n_components)
		self.columns=cols

	def fit(self,data):
		self.model.fit(data[self.columns])

	def fit_transform(self,data):
		transformed=self.model.fit_transform(data[self.columns])
		transformed=pd.DataFrame(transformed,columns=["ica_"+str(i+1) for i in range(self.n_components)])
		data=pd.concat([data,transformed],axis=1)
		data=data.drop(self.columns,axis=1)
		return data

	def transform(self,data):
		transformed=self.model.transform(data[self.columns])
		transformed=pd.DataFrame(transformed,columns=["ica_"+str(i+1) for i in range(self.n_components)])
		data=pd.concat([data,transformed],axis=1)
		data=data.drop(self.columns,axis=1)
		return data


class Ica(Preprocess):
	"""
	独立成分分析クラスです

	"""
	def __init__(self):
		super().__init__()

	def make_parser(self):
		parser=super().make_parser()
		parser.add_argument("--n_components",dest="n_components",default=2,type=int)
		#parser.add_argument("-t","--target_colname",dest="target_colname",default=None,type=str)
		return parser

	def set_parsed_args_unique(self,parsed):
		self.n_components=parsed.n_components

	def parse_args(self,args):
		parser=self.make_parser()
		return parser.parse_args(args)

	def ica(self,data):
		self.model=ica(self.columns,n_components=self.n_components)
		transformed=self.model.fit_transform(data)
		return transformed


	def main(self,args):
		parsed=self.parse_args(args)
		self.set_parsed_args_common(parsed)
		self.set_parsed_args_unique(parsed)

		data=self.read_data()
		self.columns=self.get_col_list()
		#主成分分析
		data.data=self.ica(data.data)
		#前処理をフローに追加
		data.add_preprocess(self.model)
		"""
		#変換規則のファイル出力
		with open(self.temp_files_path+"pca.pickle","wb") as f:
			pickle.dump(self.model,f)

		#前処理の順番を保存
		self.write_order()
		"""

		#独立成分データセット出力
		self.write_data(data)

if __name__=="__main__":
	ica=Ica()
	ica.main(sys.argv[1:])
