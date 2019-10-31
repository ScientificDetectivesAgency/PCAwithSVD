from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm
np.random.seed(0)

def normalData (value_1, value_2, size):
    X = np.linspace(value_1, value_2,num=size)
    #que devuelve un número aleatorio 
    # procedente de una distribución uniforme
    X0 = X*np.random.rand(len(X))+10 # Create data cluster 1
    X1 = X*np.random.rand(len(X))-10 # Create data cluster 2
    X_tot = np.stack((X0,X1,X2)).flatten()
    Normal = np.zeros((len(X_tot),2)) #Matriz para ejercio 2
    
    print('Dimensionality','=',np.shape(Normal))
    gauss_1 = norm(loc=-5,scale=5) 
    gauss_2 = norm(loc=8,scale=3)

    for c,g in zip(range(3),[gauss_1,gauss_2]):
        Normal[:,c] = g.pdf(X_tot)
    
    #Normal Dist =-Centered
    X_Normal = Normal- np.array(Normal.mean(0), ndmin=2)
    return X_Normal

#Random values for a centered matrix
X_random = np.random.rand(15,30)- np.array(X.mean(0), ndmin=2)

def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca, eigen_vals


def svd(X):
  # Data matrix X, it doesn't need to be 0-centered differently from PCA
  n, m = X.shape
  # Compute SVD
  U, Sigma, VU = np.linalg.svd(X, 
      full_matrices=False, 
      compute_uv=True)
  VUT = Vh.T # For using SVD with PCA it is necesary to transpose Vh
  # Transform X with SVD components
  X_svd = np.dot(U, np.diag(Sigma)) 
  return X_svd, VUT, Sigma

print(np.allclose(np.square(Sigma) / (n - 1), eigen_vals))

#https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
#https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
#https://intoli.com/blog/pca-and-svd/