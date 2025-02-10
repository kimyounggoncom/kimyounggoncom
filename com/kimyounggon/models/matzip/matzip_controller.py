from com.kimyounggon.models.matzip.dataset import Dataset
from com.kimyounggon.models.matzip.matzip_service import MatzipService

class MatzipController:


    dataset = Dataset()
    service = MatzipService()
    
    def modeling(self, matzip):
        this = self.dataset 
        this.matzip = self.service.new_model(matzip)
        print("🤦‍♀️맛집 데이터")
        print(this.matzip)
        return this