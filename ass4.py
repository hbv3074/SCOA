import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------
# Load dataset
# -----------------------------------
iris = pd.read_csv("SCOA_A4.csv")
# print(iris.head())
# print(iris.describe())

# Extract features and target
X = iris.drop(columns=["species"]).values
y = pd.Categorical(iris["species"]).codes   # convert species → numeric classes

# -----------------------------------
# Genetic Algorithm Setup
# -----------------------------------
POP_SIZE = 20      # number of individuals
N_GENERATIONS = 10 # iterations
MUTATION_RATE = 0.2

# Chromosome: [max_depth, min_samples_split]
def create_chromosome():
    return [random.randint(1, 20), random.randint(2, 10)]

def fitness(chromosome):
    max_depth, min_samples_split = chromosome
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

def selection(population, fitnesses):
    idx = np.argsort(fitnesses)[-2:]  # select best two
    return [population[idx[0]], population[idx[1]]]

def crossover(parent1, parent2):
    point = random.randint(0, len(parent1)-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        chromosome[0] = random.randint(1, 20)
    if random.random() < MUTATION_RATE:
        chromosome[1] = random.randint(2, 10)
    return chromosome

# -----------------------------------
# Run Genetic Algorithm
# -----------------------------------
population = [create_chromosome() for _ in range(POP_SIZE)]

for gen in range(N_GENERATIONS):
    fitnesses = [fitness(chromo) for chromo in population]
    print(f"Generation {gen} - Best Fitness: {max(fitnesses):.4f}")

    new_population = []
    parents = selection(population, fitnesses)

    for _ in range(POP_SIZE // 2):
        child1, child2 = crossover(parents[0], parents[1])
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    population = new_population

# -----------------------------------
# Best Result
# -----------------------------------
fitnesses = [fitness(chromo) for chromo in population]
best_idx = np.argmax(fitnesses)

print("\nBest Hyperparameters:", population[best_idx])
print("Best Accuracy:", fitnesses[best_idx])
