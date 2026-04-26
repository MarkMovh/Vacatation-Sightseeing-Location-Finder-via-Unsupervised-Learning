import requests
import pandas as pd
import numpy as np
import random
from math import radians, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pickle

# Genetic Algorithm (Evolutionary Computation - Not guarenteed optimal but efficient)
def generate_random_population(json_data):
    init_pop=[]

    chromosomes=list(json_data.keys())
    for i in range(1,301):
        random.shuffle(chromosomes)
        if chromosomes not in init_pop:
            init_pop.append(chromosomes.copy())
    
    return init_pop

def duration_fitness_score(chromosomes, json_data):
    overall_duration=0
    for index, chromosome in enumerate(chromosomes[:-1]):
        overall_duration += json_data[chromosome][chromosomes[index+1]][1]['value'] # Duration value

    return overall_duration


def adjust_fitness_scores(scores):
    max_fitness = max(scores)
    inverted_fitness=[(max_fitness - score) for score in scores]
    
    total_inverted_fitness=sum(inverted_fitness)
    selection_chance = [fit / total_inverted_fitness for fit in inverted_fitness]

    return selection_chance

def random_parent_selection(population,scores):
    parent = random.choices(population, weights=scores, k=1)[0]
    return parent

def fitness_diversity_selection(chromosome, population):
    # Calculate diversity based on number of different items in position
    current_candidate = ["",0]

    for parent_candidate in population:
        diversity_score = sum(1 for x, y in zip(chromosome, parent_candidate) if x != y) # Hamming distance

        if diversity_score > current_candidate[1]:
            current_candidate[0:2] = parent_candidate, diversity_score

    return current_candidate[0]

def crossover_operation(parent_1, parent_2):
    index1 = random.randint(0, len(parent_1) - 2)
    index2 = random.randint(index1 + 1, len(parent_1) - 1)

    segment = parent_1[index1:index2 + 1]
    remaining_values = [val for val in parent_2 if val not in segment]
    child_1 = remaining_values[:index1] + segment + remaining_values[index1:]

    segment = parent_2[index1:index2 + 1]
    remaining_values = [val for val in parent_1 if val not in segment]
    child_2 = remaining_values[:index1] + segment + remaining_values[index1:]

    return child_1, child_2

def mutation_operation(child):
    indices = list(range(len(child)))

    index1 = random.choice(indices)
    indices.remove(index1)
    index2 = random.choice(indices)

    child[index1], child[index2] = child[index2], child[index1]

    return child

with open('D:/School/Postgraduate/raw_path_data.pkl', 'rb') as f:
    raw_path_data = pickle.load(f)

init_rand_pop = generate_random_population(raw_path_data)
init_pop_scores = [duration_fitness_score(chromosomes, raw_path_data) for chromosomes in init_rand_pop]
avg_score = sum(init_pop_scores)/len(init_pop_scores)

avg_scores=[avg_score]
horizontal_axis=[i for i in range(0,len(avg_scores))]

plt.figure(figsize=(8, 6))
plt.ion()

population = init_rand_pop.copy()
scores = init_pop_scores.copy()
generation_count = 0

local_optimum_check = 0
best_overall_score = 0
while True:
    weighted_scores = adjust_fitness_scores(scores)
    new_population = []
    new_scores = [] 

    while len(new_population) < len(init_rand_pop):
        # Parent Selection
        parent_1 = random_parent_selection(population,weighted_scores)
        #parent_2 = random_parent_selection(population,weighted_scores)
        parent_2 = fitness_diversity_selection(parent_1, population)

        # Crossover
        crossover_probability=random.random()
        if crossover_probability > 0.6:
            child_1, child_2 = crossover_operation(parent_1,parent_2)
        else:
            child_1, child_2 = parent_1[:], parent_2[:]
        
        # Mutation
        child_1 = mutation_operation(child_1)
        child_2 = mutation_operation(child_2)

        if child_1 not in new_population and child_1 not in population:
            new_population.append(child_1)
            child_1_score = duration_fitness_score(child_1, raw_path_data)
            new_scores.append(child_1_score)
            

        if child_2 not in new_population and child_2 not in population:
            new_population.append(child_2)
            child_2_score = duration_fitness_score(child_2, raw_path_data)
            new_scores.append(child_2_score)

    top_individuals = 50
    sorted_new_population = sorted(zip(new_population, new_scores), key=lambda x: x[1])
    sorted_old_population = sorted(zip(population, scores), key=lambda x: x[1])
    best_new_individuals = sorted_new_population[:top_individuals]
    sorted_old_population[-top_individuals:] = best_new_individuals

    population, scores = zip(*sorted_old_population)
    population = list(population)
    scores = list(scores)

    avg_score = sum(scores) / len(population)
    avg_scores.append(avg_score)

    horizontal_axis.append(generation_count + 1)

    plt.clf()
    plt.plot(horizontal_axis, avg_scores, marker='o', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Average Trip Duration')
    plt.title('Average Fitness Score Over Generations (300 POP)')
    plt.draw()
    plt.pause(0.1) 

    print(f"Generation {generation_count}: Average Score: {avg_score}")
    sorted_population = sorted(zip(population, scores), key=lambda x: x[1])
    for i, (indiv, score) in enumerate(sorted_population[:3]):
        print(f"  {i + 1}. Individual: {indiv}, Duration: {score}")

    best_individual_score = sorted_population[0][1]
    if generation_count == 0:
        best_overall_score = best_individual_score
    else:
        if best_individual_score >= best_overall_score:
            local_optimum_check +=1

            if local_optimum_check == 200:
                break
        else:
            best_overall_score = best_individual_score
            local_optimum_check = 0

    generation_count += 1
    print("")
    

best_identified_path = sorted_population[0][0]
print(f'Start location: {sorted_population[0][0][0]}')
for index, location in enumerate(best_identified_path[1:]):
    if index < len(best_identified_path)-2:
        print(f'Location {index+1}: {location}')
    else:
        print(f'Final Location {index+1}: {location} - Total estimated route time: {sorted_population[0][1]/60:.2f} mins')