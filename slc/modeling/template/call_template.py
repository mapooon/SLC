from jinja2 import Template
from jinja2 import Environment, FileSystemLoader


env = Environment(loader=FileSystemLoader('./', encoding='utf8'))
template = env.get_template('model_template.py')

#モデル固有の引数や名前を設定(各変数についてはmodel_template.pyを参照)
cmd_name = "slcrf"
model_args = "min_samples_leaf=min_samples_leaf,max_depth=max_depth"
def_func_args="min_samples_leaf,max_depth"
model_name = "sklearn.ensemble.RandomForestClassifier"

model_argparse = ['"-l", "--min_samples_leaf", dest="min_samples_leaf",help="setting min_sample_leaf", default=1',
                  '"-d", "--max_depth", dest="max_depth", default=None, help="setting max_depth"']

def_model_args = ['min_samples_leaf = parsered.min_samples_leaf',
                  'max_depth = parsered.max_depth']

new_model = template.render({"cmd_name": cmd_name, "def_func_args":def_func_args,"model_args": model_args, "model_name": model_name,
                             "model_argparse": model_argparse, "def_model_args": def_model_args})

# print(new_model.encode('utf-8'))

#モデルのソースコードを生成
with open("models/"+cmd_name + ".py", "wb") as f:
    f.write(new_model.encode('utf-8'))
