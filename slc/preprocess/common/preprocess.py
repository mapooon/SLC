import sys
import argparse
import numpy as np
import pandas as pd
import os
import pickle
import dill
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../"))
from dataset import Dataset

class Preprocess():
	def __init__(self):
		#self.temp_files_path="../kcmd/preprocess/temp_files/"
		self.preprocess_list=[
						"normalize.pickle",
						"standardize.pickle",
						"label.pickle",
						"onehot.pickle",
						"pca.pickle",
						"preprocess_order.txt"
						]
	def make_parser(self):
		"""
		子クラスのmake_parser()によって内部的に呼ばれる関数です。
		共通オプションを追加したパーサーを返します。
		"""
		parser = argparse.ArgumentParser()

		parser.add_argument("-i","--input",dest="input",help="set input file name on first preprocess",default=sys.stdin,type=str)
		parser.add_argument("-o","--output",dest="output",help="set output file name if you want",default=sys.stdout,type=str)
		#parser.add_argument("-t","--temp_dir",dest="temp_dir",help="set dir to save temp files (make 'temp_files' folder under set dir)",default=os.path.join(os.getcwd(),"temp_files/"),type=str)
		parser.add_argument("--csv",dest="iscsv",help="use this option if you input csv file",action='store_const',const=True,default=False)
		parser.add_argument("-c","--columns",dest="columns",help="set column name you want to process.(with -a option, this option is ignored)",default="",type=str)
		parser.add_argument("-e","--exclude",dest="exclude",default="",type=str,help="columns except those selected by this option are preprocessed.")
		parser.add_argument("-a","--all",dest="all",action='store_const',const=True,default=False,help="use this option if you want to process all columns")
		return parser

	def set_parsed_args_common(self,parsed):
		"""
		共通のオプションを属性に追加する関数です。

		:param parsed:　コマンドライン引数をパースしたもの
		"""

		self.input=parsed.input
		self.output=parsed.output
		"""
		if not os.path.exists(parsed.temp_dir):
			os.mkdir(parsed.temp_dir)
		#self.temp_files_path=parsed.temp_dir
		"""
		self.iscsv=parsed.iscsv
		self.columns=parsed.columns
		self.all=parsed.all
		self.exclude=parsed.exclude

	def parse_args(self,args):
		"""
		コマンドライン引数をパースする関数です。
		すべてのオプションをパースしたものを返します。

		:param args: コマンドライン引数
		"""
		parser=self.make_parser()

		return parser.parse_args(args)

	def set_output(self):
		"""
		モデルの出力を設定する関数です。
		作成したモデルをファイルに書き出すか標準出力に書き込みます。
		出力先を返します。
		"""
		# ファイル書き込みかstdoutかで場合分け
		if type(self.output) == str:
			return open(self.output, 'wb')
		else:
			self.output="sys.stdout"
			return sys.stdout.buffer

	def read_data(self):
		if self.iscsv:
			data = Dataset(pd.read_csv(self.input))
		else:
			data = dill.loads(self.input.buffer.read())
		self.all_columns=list(data.data.columns)
		return data

	def write_data(self,data):
		"""
		Datasetオブジェクトを出力する関数です。
		set_outputで設定した出力先に出力します。
		"""
		#出力
		if sys.stdout.isatty():
			print(data.data)
		else:
			self.set_output().write(dill.dumps(data))
		return

	def get_col_list(self):
		"""
		処理の対象となる列名リストを返す関数です。
		"""
		columns_list=self.columns.split(",")
		exclude_list=self.exclude.split(",")
		if self.all:
			col_list=[i for i in self.all_columns if i not in exclude_list]
		else:
			col_list=[i for i in self.all_columns if i in columns_list]

		return col_list

	def write_order(self):
		"""
		前処理の順番をpreprocess_order.txtに書き込みます
		"""
		#二番目以降の処理なら追加書き込み
		if os.path.isfile(self.temp_files_path+"preprocess_order.txt"):
			with open(self.temp_files_path+"preprocess_order.txt","a") as f:
				f.write(","+self.name)
		#最初の処理なら.txtを作成して書き込み
		else:
			with open(self.temp_files_path+"preprocess_order.txt","w") as f:
				f.write(self.name)

	def remove_temp_files(self):
		#中間ファイルのフォルダがなければ作成
		if not os.path.isdir(self.temp_files_path):
			os.mkdir(self.temp_files_path)
		#中間ファイルの削除
		for filename in self.preprocess_list:
			if os.path.isfile(self.temp_files_path+filename):
				os.remove(self.temp_files_path+filename)
