import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data


#custom D biplot function
def biplot(score,coeff,labels=None,states=None):
    xs=score[:,0]
    ys=score[:,1]
    zs=score[:,2]  #extracts principal component values for each axis PC1,PC2 and PC3
    n=coeff.shape[0] #no of features

    #creating a 3D plot canvas
    fig=plt.figure(figsize=(10,12))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(xs,ys,zs,s=5) #scatter plot of observations(states) in 3D PCA space
    if states is not None:
        for i in range(len(xs)):
            ax.text(xs[i],ys[i],zs[i],states[i],size=7) #ask
        for i in range(n):
            ax.quiver(0,0,0,coeff[i,0],coeff[i,1],coeff[i,2],color='r',alpha=0.5) #ask
        labels=labels[i] if labels is not None else f"Var{i+1}"
        ax.text(coeff[i,0]*1.15,coeff[i,1],coeff[i,2]*1.15,labels,color='g',ha='center',va='center')
        #label each variable
        ax.set_xlabel("PC1")
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title("3D Biplot")
        plt.grid()
        plt.show()
def main():
    df=get_rdataset('USArrests').data
    X=df.values.squeeze()
    states=df.index #stat names are used as labels
    corr_df=df.corr() #computing correlation matrix to get variable names
    labels=corr_df.columns
    scaler=StandardScaler()
    X_std=scaler.fit_transform(X)

    pca=PCA()
    X_std_trans=pca.fit_transform(X_std)
    df_std_pca=pd.DataFrame(X_std_trans) #converting the transformed data to a dataframe
    std=df_std_pca.describe().transpose()["std"] #getting standard deviations of each PC axis

    print(f"Standard deviation:{std.values}")
    print(f"Proportion of Variance Explained:{pca.explained_variance_ratio_}")
    print(f"Cumulative Proportion: {np.cumsum(pca.explained_variance_)}")

    #calling the biplot function
    biplot(X_std_trans[:,0:3],np.transpose(pca.components_[0:3,:]),labels=list(labels),states=states)

    #feature importance
    pc1=abs(pca.components_[0])
    pc2=abs(pca.components_[1])
    pc3=abs(pca.components_[2]) #importance of each variable for PCs 1-3

    feat_df=pd.DataFrame() #df to show how much each feature contributes to each PC
    feat_df["Features"]=list(labels)
    feat_df["PC1 Importance"]=pc1
    feat_df["PC2 Importance"]=pc2
    feat_df["PC3 Importance"]=pc3
    print(feat_df)

    #cumulative variance plot
    plt.ylabel('Explained Variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_),
             c='red')
    plt.title("Cumulative Explained Variance")
    plt.show()

    #bar plot of variance explained by each individual component
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title("Scree plot")
    plt.show()

    pca_df=pd.DataFrame(X_std_trans[:,0:3],index=df.index)
    print("----------------------------------------------")
    print(pca_df.head())
if __name__=="__main__":
    main()