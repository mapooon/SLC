# What is SLC ?
SLCはコマンドラインから手軽にデータの前処理、機械学習をすることができるscikit-learnのラッパーライブラリです。

***


## Demo
SLCでは、実行したい処理をパイプラインを用いて簡単に構築できます。
ここではslc/exampleフォルダの中にあるBostonデータセットを使って、前処理から予測までの一連の流れを示します。  
Pathが通ってない場合や実行権限がない場合は各実行文の先頭に　「python」 やpythonのパス等を追記してください。  
データテーブルの拡張子は.csvでしか動作確認を行っていません。  

  ### (0)About Boston Dataset
Bostonデータセットは、米国ボストン市郊外における地域別の住宅価格のデータセットです。  
13個の説明変数と1つの目標変数(住宅価格)があり、回帰問題の例題としてしばしば用いられます。  
* train.csv : 説明変数と目標変数が入った訓練用データセットです。  
* test_pred.csv : 訓練データに含まれていないサンプルの説明変数のみが入っている、予測用データセットです。  
+ test_eval.csv : 訓練データに含まれていないサンプルの説明変数と目標変数が入っている、評価用データセットです。  

### (1)Preprocess 
ここでは標準化を行うオペレータ preprocess/standardize.py　を使います。SLCでは、パイプラインで処理をつなぐことを前提としているので(1)以外でオプション -i を指定する必要はありません。前処理オペレータでは　-c オプションで対象の列を選択します。前処理オペレータのその他のオプション等の詳しい説明はそれぞれのソースファイルを参照してください。
```
preprocess/standardize.py -c DIS -i example/boston/train.csv
```
* -c : 標準化したい列の列名(デモでは「DIS」列を標準化します)
カンマ区切りで複数の列を選択することもできます:  
```
preprocess/standardize.py -c DIS,RAD -i example/boston/train.csv
```

### (2)model training
(1)を行った後は、学習したいモデルのオペレータを実行して、モデルを学習することができます。  
デモでは、線形回帰(linear regression)で学習させます。  
```
modeling/regression/SLClinreg.py -t target -o model.pkl
```
* -t : 目的変数の列名(デモでは「target」列が目的変数列)
モデルにはそれぞれ固有のパラメータがあり、オプションでそれらを指定することもできます。オプションについてはそれぞれのモデルのソースファイルを参照してください。  

### (3)Prediction for test data
モデルを学習させた後は、試験データを指定して、そのデータの目標変数を予測することができます。このためのオペレータが postprocess/predict.py です。  
このオペレータの出力は試験データに予測ラベル列が追加されたデータテーブルになります。
```
postprocess/predict.py -i example/boston/test_pred.csv -m model.pkl -o example/boston/result.csv
```
* -i : 試験データのパス  
* -o : 出力ファイルのパス   
* -m : (2)で生成した訓練済みモデルのパス


  (1),(2)はパイプラインを用いて次のようにまとめて実行することができます。
```
preprocess/standardize.py -c DIS -i example/boston/train.csv|modeling/regression/SLClinreg.py -t target
```

---

# TODO
## Installation
gitを使って、たった一つの手順でSLCをインストールすることができます:
```
pip3 install 
```
なお、SLCの実行には以下のPythonパッケージを必要とします。  
* scikit-learn  
* numpy  
* pandas  
* pickle  
* json
* dill
 
