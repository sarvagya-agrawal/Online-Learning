'''
Author: Sarvagya Agrawal
'''



import numpy as np
import matplotlib.pyplot as plt



# Creating Data
def generate_data(N, D, rho = None):
    assert rho is None or (rho >= 0 and rho <= 1)
    mean = np.array([0,0])
    covar = np.eye(D)
    if rho is not None:
        counter = 0
        dataset = np.random.rand(N,D)
        while (counter < N):
            data = np.random.multivariate_normal(mean, covar, size = 1)
            normalized_data = data/np.linalg.norm(data)
            if np.abs(normalized_data[0,0]) > rho:
                dataset[counter] = normalized_data
                counter += 1
        return(dataset)
    else:
        data = np.random.multivariate_normal(mean, covar, size = N)
        normalized_data = np.random.rand(N, D)
        for i in range(N):
            dp = data[i,:]
            norm = np.linalg.norm(dp)
            # Normalized Data is a N x D array where each row is a data point.
            normalized_data[i,:] = dp/norm
        return(normalized_data)



x_0 = np.array([1,0])

# labelling function
def labeller(x, norm_data):
    marker = []
    y = np.random.rand(norm_data.shape[0])
    inner_prod = np.random.rand(norm_data.shape[0])
    for i in range(norm_data.shape[0]):
        inner_prod[i] = x @ norm_data[i,:]
        if inner_prod[i] > 0:
            marker.append('+')
            y[i] = 1
        else:
            marker.append('_')
            y[i] = -1
    return(y)

def symbol_plot(labellist):
    marker = []
    for i in range(labellist.shape[0]):
        if labellist[i] > 0:
            marker.append('+')
        else:
            marker.append('_')
    return(marker)


def x_star_calc(req_data, req_label, x):
    i_p = req_data[:, 0]
    y_innerprod_prod = req_label*i_p
    lambda_1 = min(y_innerprod_prod)
    x_star = x/lambda_1
   
    return x_star

# Plotting; x* shown as a blue star on Plot.
N = 40
D = 2
normalized_data = generate_data(N, D)
y= labeller(x_0, normalized_data)
marker = symbol_plot(y)
x_star = x_star_calc(normalized_data, y, x_0)
vec1 = normalized_data[:,0]
vec2 = normalized_data[:,1]
for i in range(N):
    plt.scatter(vec1[i], vec2[i], c= 'black',marker= marker[i])
plt.plot(x_star[0], x_star[1], '*')
plt.title('Exercise 3.1 Data')
plt.show()


#print(normalized_data[:,1].shape)


# loss function
def loss(label, data1, x):
    '''
    args:
    label -> (N, ) array
    data1 -> N x D
    x -> (D, ) array
    
    output:
    loss -> (N, ) array
    '''
    innerproduct = data1@x
    y_innerproduct = label*innerproduct
    term2 = 1 - y_innerproduct
    zero = np.zeros(term2.shape)
    stacked_arr = np.vstack((term2,zero))
    loss = np.amax(stacked_arr, axis = 0)
    return loss



# implementation of Online Subgradient Descent
def osd(x, data2, T, label1, xstar):
    '''
    data -> N x D array
    x -> (D, ) array
    T -> int
    label1-> (N, ) array
    xstar -> (D, ) array

    '''
    loss_vect = np.random.rand(T)
    empirical_risk_vect = np.random.rand(T)
    loss_list = []
    cumulative_loss = np.random.rand(T)
    x_tracker = [x]
    cumulative_loss_xstar = np.random.rand(T)
    xstar_losslist = []

    eta = 1/np.sqrt(T)

    for t in range(T):
        n = data2.shape[0]
        random_ind = np.random.choice(n, size = 1, replace= True)
        random_data_pt = data2[random_ind]
        random_data_pt.shape = (random_data_pt.shape[1], )
        label_rand = label1[random_ind]
        loss_t = loss(label_rand, random_data_pt, x)
        loss_vect[t] = loss_t

        loss_list.append(loss_t)
        
        if loss_vect[t] < 0:
            g_t = 0
        elif loss_vect[t] == 0:
            unif = np.random.uniform(0, 1, size = 1)
            g_t = -1*unif*random_data_pt
        else:
            g_t = -1*label_rand*random_data_pt
        x = x - eta*g_t
        x_tracker.append(x)

        # empirical risk calculation
        loss_entire_dataset = loss(label1, data2, x)
        empirical_risk = (1/N)*sum(loss_entire_dataset)
        empirical_risk_vect[t] = empirical_risk

        # Cumulative loss of SGD
        sum_till_t = sum(loss_list)
        cumulative_loss[t] = sum_till_t

        # Cumulative loss of x*
        xstar_losslist.append(loss(label_rand , random_data_pt, xstar))
        cumulative_loss_xstar[t] = sum(xstar_losslist)

    return loss_vect, empirical_risk_vect, cumulative_loss, cumulative_loss_xstar

# Calling functions and Plotting
allloss, emprisk, cumloss, cumloss_xstar = osd(np.array([0,0]), normalized_data, 200, y, x_star)
rounds = np.arange(200)

# Plotting empirical risk
plt.title('Empirical Risk as Number of rounds (T) grows')
plt.xlabel('Rounds (T)')
plt.ylabel('Empirical Risk')
plt.plot(rounds, emprisk)
plt.show()

# Plotting cumulative loss of SGD and x* on same plot
plt.title('Cumulative Loss of SGD and x*')
plt.xlabel('Rounds (T)')
plt.ylabel('Cumulative loss')
plt.plot(rounds, cumloss, label ='SGD Cumulative Loss')
plt.plot(rounds, cumloss_xstar, label ='x* cumulative loss')
plt.legend()
plt.show()

# Plotting regret
regret = cumloss - cumloss_xstar
plt.title('Regret as number of rounds grow')
plt.xlabel('Rounds (T)')
plt.ylabel('Regret')
plt.plot(rounds, regret)
plt.show()




data_size = np.arange(50, 5000, 50)
mean_3 = np.array([0,0])
covar_3 = np.eye(2)
x_o = np.array([1,0])
x_star_norm = np.random.rand(data_size.shape[0])

for i in range(data_size.shape[0]):
    x_star_temp_norm = np.random.rand(50)
    for j in range(50):
        temp_data = generate_data(data_size[i], 2)
        temp_y = labeller(x_0, temp_data)
        xstar_temp = x_star_calc(temp_data, temp_y, x_o)
        x_star_temp_norm[j] = np.linalg.norm(xstar_temp)


    x_star_norm[i] = (1/50)*sum(x_star_temp_norm)



# plotting norm of x* against N
plt.title('Norm of x* as N increases')
plt.xlabel('N (number of data pts.)')
plt.ylabel('Norm of x*')
plt.loglog(data_size, x_star_norm)
plt.show()

#creating another dataset. Setting rho as 0.3
N1 = 200
D1 = 2
rho1 = 0.3
data_4 = generate_data(N1, D1, rho1)
y_4 = labeller(x_0, data_4)
xstar4 = x_star_calc(data_4,y_4,x_0)
marker4 = symbol_plot(y_4)

print("exercise 3.4: x* = ", xstar4)

vecc1 = data_4[:,0]
vecc2 = data_4[:,1]

for i in range(N1):
    plt.scatter(vecc1[i], vecc2[i], c= 'black', marker=marker4[i])
plt.title('Exercise 3.4 Data')
plt.plot(xstar4[0], xstar4[1], '*')
plt.show()




N2 = 40
D2 = 2

rho_list = [0,0.1,0.5,0.9]
mistake_count_list = []
for i in range(len(rho_list)):
    data_6 = generate_data(N2, D2, rho_list[i])
    y_6 = labeller(x_0, data_6)
    xstar6 = x_star_calc(data_6, y_6, x_0)

    loss_rho1, _, _, _ = osd(np.array([0,0]),data_6, 200, y_6, xstar6) 
    mistake_count = np.count_nonzero(loss_rho1)
    mistake_count_list.append(mistake_count)
    print("Number of mistakes SGD makes after T = 200 rounds when N = 40, and rho = ", rho_list[i], ":", mistake_count)


