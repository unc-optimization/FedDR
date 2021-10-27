import pandas as pd

class History:
    def __init__(self, names_list):
        self.history = {}
        for name in names_list:
            self.history[name] = []
            
    def add_name(self, name):
        if name in self.history.keys():
            raise ValueError('Key already exists!')
        
        self.history[name] = []
        
    def update(self, value_list):
        for key, value in zip(self.history.keys(), value_list):
            self.history[key].append(value)
            
    def get_history(self, dataframe=True):
        if dataframe:
            return pd.DataFrame.from_dict(self.history)
        else:
            return self.history