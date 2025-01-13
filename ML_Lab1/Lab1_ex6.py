#feature data
x=[1,2,3]
#parameters
theta=[0.3,1.4,0.5]
#taret value
y=10

def hypothesis(x,theta):
    sum=[theta[i]*x[i] for i in range(len(theta))]
    print ("Hypothesis values:",sum)
    E=0
    for j in range(len(sum)):
        E=E+((sum[j]-y)**2)
    E=0.5*E
    print("Error value:",E)

def main():
    hypothesis(x,theta)

if __name__ == '__main__':
    main()





