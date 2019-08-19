import pandas as pd
import matplotlib.pyplot as plt

def calculate_cost(theta, population, profit):
    cost = 0
    for i in range(population.size):
        cost += (theta[0] + theta[1]*population[i] - profit[i])**2
    return cost/(2*population.size)

def gradient_descent(theta, population, profit, alpha):
    training_size = population.size
    derivative_theta = [0,0]
    for i in range(training_size):
        derivative_theta[0] += (theta[0] + theta[1]* population[i] - profit[i])
        derivative_theta[1] += (theta[0] + theta[1]* population[i] - profit[i])*population[i]
    return ((theta[0] - alpha*derivative_theta[0]/training_size), 
            (theta[1] - alpha*derivative_theta[1]/training_size));

dataset = pd.read_csv('data/UniVarData.txt', header = None)

population =  dataset.iloc[:,0].values
profit = dataset.iloc[:,1].values

plt.figure(0)
plt.plot(population, profit, 'ro')
plt.xlabel('Population')
plt.ylabel('profit')
plt.title('Population vs Profit')

theta = (0,0)
previous_cost = -1;
cost = calculate_cost(theta, population, profit)
cost_over_iteration = []
alpha = .01
while previous_cost != cost :
    theta = gradient_descent(theta, population, profit, alpha)
    previous_cost = cost;
    cost = calculate_cost(theta, population, profit)
    cost_over_iteration.append(cost)
    
plt.figure(1)
plt.plot(cost_over_iteration)

predictions = []
for i in range(population.size):
    predictions.append(theta[0] + theta[1]* population[i])
    
plt.figure(0)
plt.plot(population, predictions)

print ('Iterations Required:',len(cost_over_iteration) )