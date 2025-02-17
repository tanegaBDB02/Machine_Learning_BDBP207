# Implement ordinal encoding and one-hot encoding methods in Python from scratch.

class ordinal_encoder():
    def __init__(self, X):
        self.X=X

    def fit(self):
        self.sets= list(sorted(set(self.X)))
        self.dict_sets={}
        j=0
        for i in range(0, len(self.sets)):
            self.dict_sets[self.sets[i]]=j
            j+=1
        return self.dict_sets

    def transform(self, X):
        self.X=X
        for i in range(0,len(self.X)):
            self.X[i] = self.dict_sets[self.X[i]]
        return self.X

def main():
    X=['green', 'red', 'blue', 'blue', 'red']
    ord = ordinal_encoder(X)
    result = ord.fit()
    print(result)
    result = ord.transform(X)
    print(result)

if __name__ == '__main__':
    main()