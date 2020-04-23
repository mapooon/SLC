class Dataset():
    def __init__(self,df):
        self.data=df
        self.preprocess=[]
        return

    def add_preprocess(self,prop_model):
        self.preprocess.append(prop_model)
        return
