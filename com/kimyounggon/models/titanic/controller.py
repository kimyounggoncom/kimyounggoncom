from com.kimyounggon.models.titanic.dataset import Dataset
from com.kimyounggon.models.titanic.service import Service

class Controller:


    dataset = Dataset()
    service = Service()
    
    def modeling(self, train, test):
        this = self.dataset 
        this.train = self.service.new_model(train)
        print("🤦‍♀️트레인 데이터")
        print(this.train)
        this.test = self.service.new_model(test)
        print("테스트 데이터")
        return this
    
    