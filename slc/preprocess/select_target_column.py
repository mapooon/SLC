#!/usr/bin/env python3
from common.preprocess import Preprocess
import numpy as np
import pandas as pd
import sys
import pickle


class Select_target_column(Preprocess):
	def __init__(self):
		super().__init__()

	def rename_target_column(self,data):
		renamed=data.rename(columns={self.target_colname:"target"})
		return renamed

	def make_parser(self):
		parser=super().make_parser()
		parser.add_argument("-t", "--target_colname", dest="target_colname",
							default="label", type=str, help="set target_colname")
		return parser

	def set_parsed_args_unique(self,parsed):
		self.target_colname=parsed.target_colname

	def parse_args(self,args):
		"""
		コマンドライン引数をパースする関数です。
		すべてのオプションをパースしたものを返します。

		:param args: コマンドライン引数
		"""
		parser=self.make_parser()
		return parser.parse_args(args)

	def main(self,args):
		parsed=self.parse_args(args)

		self.set_parsed_args_common(parsed)
		self.set_parsed_args_unique(parsed)

		#中間ファイルの削除
		self.remove_temp_files()

		#入力ファイル読み込み
		data=pd.read_csv(self.input)
		#ターゲット列の列名を変更
		renamed=self.rename_target_column(data)

		#ターゲット列の保存
		with open(self.temp_files_path+"target_col_name.txt","w") as f:
			f.write(self.target_colname)


		renamed.to_csv(self.output,index=False)

if __name__ == '__main__':
	rename=Select_target_column()
	rename.main(sys.argv[1:])
