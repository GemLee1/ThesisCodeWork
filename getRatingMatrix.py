import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def getData (data_name, is_integer = True):
    #
    with open(
            '/Users/mine/Dropbox/Thesis/Recommender-BayesianDeep/Bayesian Deep/RecommenderSystems/citeulike-a/'+data_name+'.dat',
            'r') as f:
        next(f)  # skip first row
        data = pd.DataFrame(l.rstrip().split() for l in f)
    dims = data.shape
    data_ar = data.as_matrix()
    data_ar = np.reshape(data_ar, dims[0]*dims[1])
    data_vals = data_ar
    if (is_integer == True):
        data_vals = [int(w or -1) for w in data_ar]
    data_ar = np.reshape(data_vals,dims)
    return(data_ar)

def user_data_to_rating_matrix(user_data):
    dims = np.shape(user_data)
    rating_mat = np.reshape(np.zeros(dims[0]*(np.amax(user_data)+1)),(dims[0],(np.amax(user_data)+1)))
    for u in range(0, dims[0]-1):
        for i in user_data[u,:]:
            if (i != -1):
                rating_mat[u, i] = 1
    return(rating_mat)

def reduce_data(data, x, y):
    return(data[0:x,0:y])

def get_initialization(dim_x,dim_y,mean,variance):
    return(np.random.normal(mean,variance,(dim_x,dim_y)))

def gradient(d, i, j, U, V, R, I, var, var_u, var_v, which):
    if (which==1):
        M = np.shape(V)[1]
        tot = 0
        for j in np.arange(0,M-1,1):
            tot += I[i][j]*(V[d][j] ** 2*U[d][i]-R[i][j]*V[d][j])
        return(tot+var*U[d][i]/var_u)
    else:
        N = np.shape(U)[1]
        tot = 0
        for i in np.arange(0, N - 1, 1):
            tot += I[i][j] * (U[d][i] ** 2 * V[d][j] - R[i][j]*U[d][i])
        return (tot + var * V[d][j] / var_v)

def minimization_of_objective(N, M, D, R, I, var, var_u, var_v, step_size, precision):
    U = get_initialization(D, N, 0, var_u)
    V = get_initialization(D, M, 0, var_v)
    j=0

    for t in np.arange(0,70):
        for d in np.arange(0,D-1):
            for i in np.arange(0,N-1,1):
                old_u = U[d][i]
                U[d][i] += - step_size * gradient(d, i, j, U, V, R, I, var, var_u, var_v, 1)
            for j in np.arange(0, M-1, 1):
                old_v = V[d][j]
                V[d][j] += - step_size * gradient(d, i, j, U, V, R, I, var, var_u, var_v, 0)

    return(U,V)

def get_precision(R, U, V, var):
    (N,M) = np.shape(R)
    pres = 0
    for i in range(0,N-1):
        for j in range(0,M-1):
            g_ij = expit(np.matmul(np.matrix.transpose(U[:,i]),V[:,j]))
            guess = np.random.normal(g_ij, var)
            #rnd = np.random.uniform()
            if (guess >= 0.5 and R[i,j]==1):
                pres += 1
    return(pres/sum(sum(R)))


def get_training_data(R, P):
    (N, M) = np.shape(R)
    I = np.zeros((N,M))
    for i in range(0,N-1):
        indx = np.argwhere(R[i,:]>0)
        ln = indx.size
        if (ln <= P):
            I[i,:] = R[i,:]
        else:
            choice = np.random.choice(range(indx.size), P, False)
            I[0, choice] = 1
    return(I)


data = getData('users')
mat = user_data_to_rating_matrix(data)
R = reduce_data(mat, 30, 500)

D = 20
var = 1
var_u = 1
var_v = 1
step_size = 0.0001
precision = 0.000001
(N,M) = np.shape(R)
I = R


I = get_training_data(R, 5)
(U,V) = minimization_of_objective(N, M, D, R, I, var, var_u, var_v, step_size, precision)

#print(U)
#print(V)
#print(R)
#print(np.matmul(np.matrix.transpose(U),V))


print(get_precision(R,U,V,var))


#plt.matshow(red_data+1)
#plt.show()


