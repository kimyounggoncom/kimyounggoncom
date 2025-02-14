from com.kimyounggon.models.titanic.dataset import Dataset
import pandas as pd

"""
PassengerId  고객ID,
Survived 생존여부,
Pclass 승선권 1 = 1등석, 2 = 2등석, 3 = 3등석,
Name,
Sex,
Age,
SibSp 동반한 형제, 자매, 배우자,
Parch 동반한 부모, 자식,
Ticket 티켓번호,
Fare 요금,
Cabin 객실번호,
Embarked 승선한 항구명 C = 쉐브루, Q = 퀸즈타운, S = 사우스햄튼
print(f'결정트리 활용한 검증 정확도 {None}')
print(f'랜덤포레스트 활용한 검증 정확도 {None}')
print(f'나이브베이즈 활용한 검증 정확도 {None}')
print(f'KNN 활용한 검증 정확도 {None}')
print(f'SVM 활용한 검증 정확도 {None}')
# """
class TitanicService:

    dataset = Dataset()


    def new_model(self, fname) -> object:
        this = self.dataset 
        this.context = 'C:\\Users\\bitcamp\\Documents\\project250207\\com\\kimyounggon\\datas\\titanic\\'
        this.fname = fname
        return pd.read_csv(this.context + this.fname)



    @staticmethod
    def pclass_ordinal():
        pass

    @staticmethod
    def gender_ordinal():
        pass

    @staticmethod
    def age_ordinal():
        pass

    @staticmethod
    def fare_ordinal():
        pass

    @staticmethod
    def Embarked_nominal():
        pass
    