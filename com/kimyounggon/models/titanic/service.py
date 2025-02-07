from com.kimyounggon.models.titanic.dataset import Dataset
import pandas as pd
class Service:

    dataset = Dataset()


    def new_model(self, fname) -> object:
        this = self.dataset 
        this.context = 'C:\\Users\\bitcamp\\Documents\\project250207\\com\\kimyounggon\\datas\\titanic\\'
        this.fname = fname
        return pd.read_csv(this.context + this.fname)




    