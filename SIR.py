import numpy as np
import scipy
from scipy.special import betaln
from scipy.special import psi, polygamma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import beta
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.special import betaln, digamma

# Load the data
file_path = r'C:\Users\chado\Downloads\time-series-19-covid-combined.csv'
data = pd.read_csv(file_path)

#Process Data
def preprocessdata(data):
    data = np.array(data)
    data = data[data != 0]                  #Get rid of zeros
    return data

rows = data.iloc[150538:150705]              #Lines for the time interval

subset_rows = data.iloc[150538:150705]
confirmed_values = subset_rows['Confirmed'].values
confirmed = confirmed_values.tolist()
confirmed_changes = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
confirmed_changes = [x / 128000000 for x in confirmed_changes]
confirmed2 = preprocessdata(confirmed_changes)

recovered_values = subset_rows['Recovered'].values
recovered = recovered_values.tolist()
recovered_changes = [recovered[i] - recovered[i - 1] for i in range(1, len(recovered))]
recovered_changes = [x / 128000000 for x in recovered_changes]
recovered2 = preprocessdata(recovered_changes)                                         #Change to numpy array


#Find MLES
def fisher_scoring(data, tol=0.001, max_iter=100):
    update = np.array([.1, .1])
    for i in range(max_iter):
        alpha, beta = update[0], update[1]

        score_alpha = np.sum(np.log(data) - digamma(update[0]) + digamma(update[0] + update[1]))
        score_beta = np.sum(np.log(1 - data) - digamma(update[1]) + digamma(update[0] +update[1]))
        score = np.array([score_alpha, score_beta])                 #Find Score Funtion

        info_alpha = polygamma(1, update[0]) + polygamma(1, update[0] + beta)
        info_beta = polygamma(1, beta) + polygamma(1, update[0] + beta)
        info_alphabeta = polygamma(1, update[0] + beta)
        info = np.array([[info_alpha, info_alphabeta],[info_alphabeta, info_beta]])#Find information matrix

        old = update
        update = update + np.linalg.solve(info, score)    #Update fisher scoring
        if np.linalg.norm(update - old) / np.linalg.norm(update) < tol:       #Check for convergance
            return update[0], update[1]
    return


alpha_confirmed, beta_confirmed = fisher_scoring(confirmed2)
print(alpha_confirmed, beta_confirmed)
alpha_recovered, beta_recovered = fisher_scoring(recovered2)
print(alpha_recovered, beta_recovered)


# Initial conditions
S0 = 0.98
I0 = 0.01
R0 = 0.01

# Time parameters
T = 365 # total time steps
dt = 1  # time step

S = np.zeros(T)
I = np.zeros(T)
R = np.zeros(T)
S[0], I[0], R[0] = S0, I0, R0

#Eulers Method
for t in range(1, T):
    beta1 = np.random.beta(alpha_confirmed, beta_confirmed)
    gamma1 = np.random.beta(alpha_recovered, beta_recovered)

    S[t] = S[t-1] + (-beta1 * S[t-1] * I[t-1] ) * dt
    I[t] = I[t-1] + (beta1 * S[t-1] * I[t-1] - gamma1 * I[t-1] ) * dt
    R[t] = R[t-1] + (gamma1 * I[t-1] ) * dt

#Plot trajectories
plt.plot(S, label="S(t)")
plt.plot(I, label="I(T)")
plt.plot(R, label="R(t)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.show()



# Simulate and make a histogram
percentage = []
for i in range(1, 100):
    S = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)
    S[0], I[0], R[0] = S0, I0, R0
    for t in range(1, T):
        beta1 = np.random.beta(alpha_confirmed, beta_confirmed)
        gamma1 = np.random.beta(alpha_recovered, beta_recovered)

        S[t] = S[t - 1] + (-beta1 * S[t - 1] * I[t - 1]) * dt
        I[t] = I[t - 1] + (beta1 * S[t - 1] * I[t - 1] - gamma1 * I[t - 1]) * dt
        R[t] = R[t - 1] + (gamma1 * I[t - 1]) * dt
        percentage.append(0.98 - min(S))

plt.hist(percentage, bins=20, color='blue')
plt.xlabel('Percentage of Population Infected')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()
