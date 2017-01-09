import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def getData (data_name, is_numerical = True):
    # data_name : name of the data file 
    # is_integer: True if the data is numerical
    # Reads the data on the file to DataFrame and converts it to ndarray
    # Data is assumed to be two dimensional
    with open(
            './citeulike-a/'+data_name+'.dat',
            'r') as f:
        next(f)  # skip first row
        data = pd.DataFrame(l.rstrip().split() for l in f)
    dims = data.shape
    data_ar = data.as_matrix()
    data_ar = np.reshape(data_ar, dims[0]*dims[1])
    data_vals = data_ar
    if (is_numerical == True):
        data_vals = [int(w or -1) for w in data_ar]
    data_ar = np.reshape(data_vals,dims)
    return(data_ar)

def user_data_to_rating_matrix(user_data):
    # user_data : array of the user rating data
    # Returns a binary matrix, where x axis is the user and y axis is the item dimension
    # Value is 1 if the user preference on the corresponding item
    
    dims = np.shape(user_data)
    rating_mat = np.reshape(np.zeros(dims[0]*(np.amax(user_data)+1)),(dims[0],(np.amax(user_data)+1)))
    for u in range(0, dims[0]-1):
        for i in user_data[u,:]:
            if (i != -1):
                rating_mat[u, i] = 1
    return(rating_mat)

def reduce_data(data, x, y):
    # retruns a smaller version of the data x and y dimensions particularly 
    return(data[0:x,0:y])

def get_initialization(dim_x,dim_y,mean,variance):
    # initialization of the parameters 
    return(np.random.normal(mean,variance,(dim_x,dim_y)))

def gradient(d, i, j, U, V, R, I, var, var_u, var_v, which):
    # d, i and j:  are the current idexes
    # U: D X N user feature matrix
    # V: D x M item feature matrix
    # R: main rating matrix
    # I: mask matrix, binary matrix, currently known values are 1
    # var, var_u, var_v : variances 
    # which: 1 if the current gradient is for the user feature matrix, o otherwise
    
 
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

def minimization_of_objective(N, M, D, R, I, var, var_u, var_v, step_size, precision, num_iter):
    U = get_initialization(D, N, 0, var_u)
    V = get_initialization(D, M, 0, var_v)
    j=0

    for t in np.arange(0,num_iter):
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
    Rec_mat = np.matmul(np.matrix.transpose(U), V)
    for i in range(0,N-1):
        for j in range(0,M-1):
            #g_ij = expit(np.matmul(np.matrix.transpose(U[:,i]),V[:,j]))
            g_ij = Rec_mat[i,j]
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

def get_M_recommendation(U, V, M):
    N = np.shape(U)[1]
    #M = np.shape(V)[1]

    Recm_vals =  np.matmul(np.matrix.transpose(U), V)
    Recm = np.zeros((N,M))
    for i in range(0,N):
        pred_list = Recm_vals[i,]
        rec_ind = np.argsort(pred_list)
        Recm[i,] = rec_ind[0:M]

    return(Recm)

  
def recall_M(R, U, V, M, data):
    N = np.shape(U)[1]
    Recm = get_M_recommendation(U, V, M)
    non_zero_R = np.nonzero(R)
    indexes = np.array(non_zero_R)
    recall_m = 0
    
    for i in range(0,N):
        user_choices = indexes[1,np.where(indexes[0,]==i)]
        recommended = Recm[i,]
        intersect = np.intersect1d(user_choices,recommended)
        recall_m += len(intersect)/len(user_choices)
        
    return(recall_m / N)
        
    
def mAP_mean_average_precision(R, U, V, M):
    N = np.shape(U)[1]
    Recm = get_M_recommendation(U, V, M)
    non_zero_R = np.nonzero(R)
    indexes = np.array(non_zero_R)
    ap_vals = np.zeros((N,1))
    print(indexes)
    
    for i in range(0,N):
        prob_k = 0
        so_far = 0
        user_choices = indexes[1,np.where(indexes[0,]==i)]
        if empty(user_choices):
            ap_vals[i] = 0
            continue
        for k in range(0,M):
            if (np.nonzero(user_choices-Recm[i,k]==0)[0]==0):
                so_far +=1
                prob_k += so_far/(k+1)
        ap_vals[i] = prob_k/min(len(user_choices[0,]),M)
    return(ap_vals)
                
def empty(seq):
     try:
         return all(map(empty, seq))
     except TypeError:
         return False
    
    

#data = getData('users')
#mat = user_data_to_rating_matrix(data)
#R = reduce_data(mat, 30, 500)
#
#D = 25
#var = 1
#var_u = 1
#var_v = 1
#step_size = 0.0001
#precision = 0.000001
#num_iter = 100
#(N,M) = np.shape(R)
#I = R
#
#
#I = get_training_data(R, 5)
#(U,V) = minimization_of_objective(N, M, D, R, I, var, var_u, var_v, step_size, precision, num_iter)

#print(U)
#print(V)
#print(R)
#print(np.matmul(np.matrix.transpose(U),V))


#print(get_precision(R,U,V,var))

#Recm = get_M_recommendation(U, V, 50)
#print(recall_M(R, U, V, 50, data))
#print(mAP_mean_average_precision(R, U, V, 100))


#plt.matshow(red_data+1)
#plt.show()

R_t = [[ 1.,  0.,  0.,  1.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  1.,  0.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  1.,  0.]]
U_t = U[0:10,0:5]
V_t = V[0:10,0:7]


print(get_M_recommendation(U_t, V_t, 3))
print(mAP_mean_average_precision(R_t, U_t, V_t, 3))

