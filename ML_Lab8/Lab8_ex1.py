def L2_norm(vector):
    sum=0
    for i in vector:
        sum+=i*i
    return sum

def L1_norm(vector):
    sum=0
    for i in vector:
        sum+= abs(i)
    return sum

def main():
    A=[1,5,4,7,9,0]
    print("L1: ",L1_norm(A))
    print("L2: ",L2_norm(A))

if __name__=="__main__":
    main()