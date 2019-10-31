from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm
np.random.seed(0)


#La función numpy.linspace genera un 
# array NumPy formado por n números equiespaciados
#numpy.linspace(valor-inicial, valor-final, número de valores)

def genrateData (initialize_value, last_value, number_ofvalues): 
    """Genera tres distribusiones normales de datos, y asigna probabilidades en funcion de esas distribuciones"""
    X = np.linspace(-10, 10,num=20)
    #que devuelve un número aleatorio 
    # procedente de una distribución uniforme
    X0 = X*np.random.rand(len(X))+10 # Create data cluster 1
    X1 = X*np.random.rand(len(X))-10 # Create data cluster 2
    X2 = X*np.random.rand(len(X)) # Create data cluster 3
    X_tot = np.stack((X0,X1,X2)).flatten() # Combina los clusters

    #Genera un array con una dimensionalidad NxK donde N es el numero de elementos por por cluster y K es el numero de clusters
    r = np.zeros((len(X_tot),3))  
    print('Dimensionality','=',np.shape(r))

    # En esta parte se dan valores a las gaussianas loc=media, scale=Standard desviation probability density function = norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
    gauss_1 = norm(loc=-5,scale=5) 
    gauss_2 = norm(loc=8,scale=3)
    gauss_3 = norm(loc=1.5,scale=1)
    for c,g in zip(range(3),[gauss_1,gauss_2,gauss_3]):
        r[:,c] = g.pdf(X_tot) # ahora la matriz r están en funcion de los valores de cada gaussiana 
    
    for i in range(len(r)):
        r[i] = r[i]/np.sum(r,axis=1)[i] #Esta parte incluye la probabilidad de que un valor X pertenezaca a una gaussiana S, la probabilidad de que este punto de 
                                        #datos pertenezca a una gaussiana dividido por la suma de las probabilidades.
    print(r)
    print(np.sum(r,axis=1))    
    return r

genrateData(-10, 10, 20)


number_Ofclusters = [1, 2 ,3] 

lst = []
for  i in number_Ofclusters:
    line = np.linspace(-10, 10,num=20)*np.random.rand(len(X))+10
    lst.append(line)


X = np.linspace(-10, 10,num=20)
X_tot = np.stack(((X*np.random.rand(len(X))+10),(X*np.random.rand(len(X))-10),X*np.random.rand(len(X)))).flatten()

array = np.arange(20)