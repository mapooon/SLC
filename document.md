# About SLC
SLCは preprocess(前処理), modeling(モデル作成), postprocess(後処理) の3種類のオペレータがあります。  


***


## preprocess
preprocessオペレータはデータを整形・加工するために用いられます。  
SLCでは以下のpreprocessオペレータが実装されています。  
* factor_analysis  : 因子分析
* ica : 独立成分分析
* kbins_discretizer : Kビン分割
* kernel_centerer : カーネル中央化
* leave_one_out_encode : leave one out エンコード
* multi_label_binarizer : マルチラベル二値化
* normalize : 正規化
* one_hot_encode : one hot エンコード
* ordinal_encode : ラベルエンコード
* pca : 主成分分析
* robust_scaler : ロバストスケール化
* sparce_pca : スパース主成分分析
* standardize : 標準化
***

## modeling
modelingオペレータは機械学習モデルを作成し、訓練するために用いられます。  
modelingオペレータはさらにclassification, regression, clusterの3種類に分けられます。
* classification
    * kab : adaboost
    * kbag : bagging
    * kdt : decision tree
    * kgaussian_nb : gaussian naive bayes
    * kgb : gradient boost
    * klogreg : logistic regression
    * knearest_neighbor : nearest neighbor
    * kneuralnet : neural network
    * krf : random forest
    * ksvm : support vector machine
* regression
    * kab : adaboost
    * kbag : bagging
    * kdt : decision tree
    * kelastic : elastic net
    * kgb : gradient boost
    * klasso : lasso regression
    * klinreg : linear regression
    * knearest_neighbor : nearest neighbor
    * kneuralnet : neural network
    * krf : random forest
    * kridge : ridge regression
    * ksvm : support vector machine
* cluster
    * kkmeans : k-means
***

## postprocess
postprocessオペレータは学習したモデルを使って未知データの予測をするために用いたり、精度を評価するために用いられます。
* predict : 入力された説明変数に対する予測を出力します
* evaluate : 入力された説明変数と目標変数を使って訓練済みモデルを評価します