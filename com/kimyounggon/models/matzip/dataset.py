from dataclasses import dataclass 

@dataclass
class Dataset:
    matzip: object
    context: str
    fname: str
    label: str
    


    @property
    def matzip(self) -> object:
        return self._matzip 
    
    @matzip.setter
    def matzip(self, matzip):
        self._matzip = matzip 

    @property
    def context(self) -> str:
        return self._context 
    
    @context.setter
    def context(self, context):
        self._context = context 

    @property
    def fname(self) -> str:
        return self._fname 
    
    @fname.setter
    def fname(self, fname):
        self._fname = fname 

    @property
    def label(self) -> str:
        return self._label 
    
    @label.setter
    def label(self, label):
        self._label = label 

    @property
    def label(self) -> str:
        return self._label 
    
    @label.setter
    def label(self, label):
        self._label = label        
