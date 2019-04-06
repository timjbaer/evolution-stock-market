'''
obtain and format data (stock information for 38 tech companies over last 3 months, ending yesterday)
'''

import requests
import random
import numpy as np

TOKEN = "pk_2834cb48ae0144a0a4ae1d23717770d8" # edit if using your own IEX account
SYMBOLS = """abde,amd,googl,goog,adi,aapl,amat,asml,adsk,bidu,
    avgo,cdns,cern,chkp,csco,ctxs,fb,intc,intu,klac,
    lrcx,mxim,mchp,mu,msft,ntap,ntes,nvda,nxpi,qcom,
    swks,symc,snps,txn,vrsn,wdc,wday,xlnx"""
TYPES = "chart"
RANGE = "3m"

query = {"token": TOKEN, "symbols": SYMBOLS, "types": TYPES, "range": RANGE}
r = requests.get("https://cloud.iexapis.com/beta/stock/market/batch", params=query)

dict = r.json()

NUM_DAYS = 50 # a little less than 3 months
NUM_TRAINING_DAYS = 40

training_data = np.array([0, 0, 0, 0, 0]) # python is weird
testing_data = np.array([0, 0, 0, 0, 0])
    
for symbol in dict:
    for i in range(NUM_DAYS):
        openPrice = dict[symbol]["chart"][i]["open"] # open is a keyword
        high = dict[symbol]["chart"][i]["high"]
        low = dict[symbol]["chart"][i]["low"]
        close = dict[symbol]["chart"][i]["close"]
        volume = dict[symbol]["chart"][i]["volume"]
        predicted_price = dict[symbol]["chart"][i + 7]["open"]

        if i < NUM_TRAINING_DAYS:
            training_data = np.vstack([training_data, [openPrice, high, low, close, predicted_price]])
        else:
            testing_data = np.vstack([training_data, [openPrice, high, low, close, predicted_price]])

    
training_data = training_data[1::]
testing_data = testing_data[1::]

'''
Genetic Algorithm for Stock Market Data
Credits to MorvanZhou: https://github.com/yuanlairucisky/MorvanZhou-Evolutionary-Algorithm
'''
class Genetic_Algorithm:


    # Get stock training data
    def __init__(self, train_data, DNA_SIZE = 10, POP_SIZE = 100, CROSS_RATE = 0.7, MUTATION_RATE = 0.2, N_GENERATIONS = 100):
        #Define constants for algorithm
        self.DNA_SIZE = DNA_SIZE # Number of input variables
        self.POP_SIZE = POP_SIZE # Number of models in current generation
        self.CROSS_RATE = CROSS_RATE # Probability of crossover event
        self.MUTATION_RATE = MUTATION_RATE # Probability of a mutation occuring
        self.N_GENERATIONS = N_GENERATIONS # Number of generations in algorithm
        self.input_data = train_data[:,0:DNA_SIZE]
        self.output_data = train_data[:,DNA_SIZE:DNA_SIZE + 1]
        self.NUM_EXAMPLES = self.input_data.shape[0]

    # Find fitness for selection
    def get_fitness_one(self,calculated_out, real_out):
        reciporocal_reals = 1. / real_out
        reciporocal_out = 1. / calculated_out
        averaging_factor = 1. / self.NUM_EXAMPLES
        fitness_array = np.absolute(1 - (averaging_factor * np.matmul(calculated_out, reciporocal_reals)))
        fitness_array_two = np.absolute(1 - (averaging_factor * np.matmul(reciporocal_out, real_out)))
        return np.sqrt((fitness_array * fitness_array_two))
    def get_fitness_two(self,calculated_out,real_out): #DO NOT USE THIS FUNCTION EVER
        check_out = real_out
        fitness_array = np.zeros(len(calculated_out),dtype=np.dtype('f'))
        for i in range(0,len(calculated_out)):
            to_check = calculated_out[i]
            mse = (abs(to_check - check_out)).mean(axis=None)
            fitness_array[i] = mse
        #fitness_array = np.transpose(fitness_array)
        return fitness_array
        
    
    # Get predicted values of models
    def translateDNA(self,pop):
        transposed_input = np.transpose(self.input_data)
        return np.matmul(pop,transposed_input)

    # Get next generation based on fitness values
    def select(self,pop, fitness):
        fitness_array = np.transpose(fitness)
        selected_indexes = fitness_array.argsort()[0:self.POP_SIZE]
        selected_pop = pop[selected_indexes]
        return selected_pop

    # Crossover process
    def crossover(self,parent, pop):
        if np.random.rand() < self.CROSS_RATE: # Ensures crossover happens at crossover rate
            i_ = np.random.randint(0, self.POP_SIZE) # Select another individual from pop
            cross_points = np.random.randint(0, 2, size=self.DNA_SIZE).astype(np.bool) # Choose crossover points
            parent[cross_points] = pop[i_,cross_points] # Mating and produce one child
        return parent


    def mutate(self,pop): # Mutation Process
        for index in range(0,len(pop)):
            for point in range(0,self.DNA_SIZE):
                if np.random.rand() < self.MUTATION_RATE: # Ensures mutation happens at mutation rate
                    if np.random.rand() < 0.5:
                        pop[index,point] += 0.0001
                    else:
                        pop[index,point] -= 0.0001
        return pop

factorArr = 1
num = 0
modelList = []
fitnessList = []
factorList = []
while(num < 50):
    pop = abs(np.random.randn(100, training_data.shape[1] - 1).astype('f'))*factorArr #initialize random population
    ga = Genetic_Algorithm(training_data, DNA_SIZE = training_data.shape[1] - 1)
    mostFit = 0
    for i in range(0,ga.N_GENERATIONS):
        pop = ga.mutate(pop)
        pop_copy = pop.copy()
        for j in range(0,pop_copy.shape[0]):
            np.vstack([pop,ga.crossover(pop_copy[j,:],pop_copy)[None,:]])
        calculated_results = ga.translateDNA(pop)
        fitness = ga.get_fitness_one(calculated_results,ga.output_data)
        #print(min(fitness))
        #print("Most fit model:",pop[np.argmin(fitness),:])
        pop = ga.select(pop,fitness)
        pop = pop[0]
        if i == ga.N_GENERATIONS - 1:
            mostFit = pop[np.argmin(fitness),:]
    modelList.append(mostFit)
    testGa = Genetic_Algorithm(testing_data, DNA_SIZE = testing_data.shape[1] - 1, POP_SIZE = 1)
    calculated_results = testGa.translateDNA(mostFit)
    fitness = testGa.get_fitness_one(calculated_results,testGa.output_data)
    fitnessList.append(fitness[0])
    factorList.append(factorArr)
    print(fitness)
    factorArr = mostFit
    num += 1
print()
print()
smallest_val = min(fitnessList)
arg=fitnessList.index(smallest_val)
best_factor = factorList[arg]
best_model = modelList[arg]
print("Smallest Fitness value:",smallest_val)
print("Final Factor Value:",best_factor)
print("Best Model:",best_model)




SYMBOLS = "aapl"
SYMBOLS_UPPER="AAPL"
RANGE = "1d"



query = {"token": TOKEN, "symbols": SYMBOLS, "types": TYPES, "range": RANGE}

r = requests.get("https://cloud.iexapis.com/beta/stock/market/batch", params=query)



realtime = r.json()
#print(realtime)


aapl_realtime = np.array([realtime[SYMBOLS_UPPER]["chart"][0]["average"], realtime[SYMBOLS_UPPER]["chart"][0]["high"], realtime[SYMBOLS_UPPER]["chart"][0]["low"], realtime[SYMBOLS_UPPER]["chart"][0]["average"]], dtype=np.dtype('f'))

predicted_price = np.matmul(best_model,np.transpose(aapl_realtime))



print("aapl price 7 days from now:", predicted_price)
