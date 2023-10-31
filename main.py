import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import copy
import random
import numpy as np
from plot import plot_surface,  title_f1, title_f2, plot_contour, plot_contour_with_two_populations
from functions import DOMAIN, get_f1, get_f2 



Genome = List[int]
class Individual:
	# genome[0] = x1
	# genome[1] = y1
	# genome[2] = x2
	# genome[3] = y2
	def __init__(self, genome: Genome):
		self.genome = genome

	def __str__(self):
		string_genome = [str(round(gene, 5)) for gene in self.genome]
		return '[' + ', '.join(string_genome) + ']'
		
Population = List[Individual]


def get_cost(individual: Individual) -> float:
	return get_f1(individual.genome[0], individual.genome[1]) + get_f2(individual.genome[2], individual.genome[3])	

def get_cost_population(population: Population) -> float:
	return sum([get_cost(genome) for genome in population])

def sort_population(population: Population) -> Population:
	return sorted(population, key=get_cost)

def get_best_individual(population: Population) -> Individual:
	sorted_pop = sort_population(population)
	return sorted_pop[0]


def get_random_population(number_individuals = 20, domain = DOMAIN) -> Population:
	"""
	returns a population considering constraints of domain
	"""
	population: Population = []
	for _ in range(number_individuals):
		genome = get_random_genome(domain)
		individual = Individual(genome)
		population.append(individual)
	return population


def get_random_genome(domain = DOMAIN) -> Genome:
	"""
	returns a genome considering constraints of domain
	"""
	genome: Genome = []
	for i in range(4):
		if i % 2 == 0:
			genome.append(random.uniform(DOMAIN.x_lower_bound, DOMAIN.x_upper_bound))
		else:
			genome.append(random.uniform(DOMAIN.y_lower_bound, DOMAIN.y_upper_bound))
	return genome

def get_random_genome_gauss():
	pass

def tournament_selection(population: Population) -> Population:
	next_population: Population = []
	for _ in range(len(population)):
		individual_1 = random.choice(population) 
		individual_2 = random.choice(population) 
		if get_cost(individual_1) > get_cost(individual_2):
			next_population.append(individual_2)
		else:
			next_population.append(individual_1)
	return next_population


def single_point_crossing(individual_1: Individual, individual_2: Individual) -> Tuple[Individual, Individual]:
	if len(individual_1.genome) != len(individual_2.genome):
		raise ValueError("genomes should have the same length!")

	point_index = random.randint(1, len(individual_1.genome) - 1)
	
	new_genome_1 =  individual_1.genome[0:point_index] + individual_2.genome[point_index:]
	new_genome_2 =  individual_2.genome[0:point_index] + individual_1.genome[point_index:]
 
	return Individual(new_genome_1), Individual(new_genome_2) 



def single_point_crossing_population(population: Population, crossing_probability: float)-> Population:
	"""
	crossing_probability should have value between [0, 1]
	"""
	current_population_copy = [copy.deepcopy(individual) for individual in population]
	new_population: Population = []
	while len(current_population_copy) > 1:
		rand_number = random.uniform(0, 1)
		index_parent_1 = random.randint(0, len(current_population_copy)-1)
		parent_1 = current_population_copy[index_parent_1]
		current_population_copy.pop(index_parent_1)
		index_parent_2 = random.randint(0, len(current_population_copy)-1)
		parent_2 = current_population_copy[index_parent_2]
		current_population_copy.pop(index_parent_2)
		if crossing_probability == 1 or rand_number < crossing_probability:
			offspring_1, offspring_2 = single_point_crossing(parent_1, parent_2)
			new_population.append(offspring_1)
			new_population.append(offspring_2)
		else:
			new_population.append(parent_1)
			new_population.append(parent_2)
	if len(current_population_copy) == 1:
		new_population.append(current_population_copy[0])
		current_population_copy.pop()
	return new_population


def gaussian_mutation(individual: Individual, mutation_probability: float, mutation_power: float):
	for i in range(len(individual.genome)):
		rand_number = random.uniform(0, 1)
		if mutation_probability == 1 or rand_number < mutation_probability:
			mutation_offset = mutation_power*np.random.normal(0, 1)
			individual.genome[i] += mutation_offset


def gaussian_mutation_population(population: Population, mutation_probability: float, mutation_power: float) -> Population:
	"""
	mutation_probability should have value between [0, 1]
	"""
	new_population = [copy.deepcopy(individual) for individual in population]
	for individual in new_population:
		gaussian_mutation(individual, mutation_probability, mutation_power)
	
	return new_population


# def run_simulation(
# 		cost_population_func: Callable[[Population], float],
# 		cost_individual_func: Callable[[Individual], float],
# 		get_best_individual: Callable[[Population], Individual],
# 		get_init_population: Callable[[int], Population],
# 		selection_func: Callable[[Population], Population],
# 		crossing_func: Callable[[Population, float], Population],
# 		mutation_func: Callable[[Population, float, float], Population],
# 		succession_func: Callable[[Population], Population] = None,
# 		crossing_probability: float = 0.9,
# 		mutation_probability: float = 0.2,
# 		mutation_power: float = 5,
# 		number_individual: int = 100,
# 		max_iteration: int = 100

# ) -> Tuple[Individual, float]:
# 	populations: List[Population] = []
# 	init_population = get_init_population(number_individual)
# 	best_population_cost = init_population_cost = cost_population_func(init_population)
# 	populations.append(init_population)
# 	best_individual = get_best_individual(init_population)
# 	best_individual_cost = cost_individual_func(best_individual)
# 	curr_population = init_population
# 	best_population = init_population
# 	##########################
# 	for _ in range(max_iteration):
# 		curr_population = selection_func(curr_population)
# 		curr_population = crossing_func(curr_population, crossing_probability)
# 		curr_population = mutation_func(curr_population, mutation_probability, mutation_power)
# 		curr_population_cost = cost_population_func(curr_population)
# 		if best_population_cost > curr_population_cost:
# 			best_population = curr_population
# 		curr_best_individual = get_best_individual(curr_population)
# 		curr_best_individual_cost = cost_individual_func(curr_best_individual)

# 		if curr_best_individual_cost < best_individual_cost:
# 			best_individual = curr_best_individual
# 			best_individual_cost = curr_best_individual_cost
# 		if succession_func is not None:
# 			curr_population = succession_func(curr_population)
# 		populations.append(curr_population)
# 		# print(curr_best_individual_cost)
# 	print('--------')
# 	# print(get_cost(get_best_individual(init_population)))
# 	# print(get_cost(best_individual))
# 	# for population in populations:
		
# 	# 	curr_best_individual = get_best_individual(population)
# 	# 	curr_best_individual_cost = get_cost(curr_best_individual)
# 	# 	print(curr_best_individual_cost)
# 	for individual in init_population:
# 		print(individual)
# 	print('--------')
# 	print(best_individual)
# 	print(best_individual_cost)
# 	return (best_individual, best_individual_cost)


def run_simulation(
		cost_population_func: Callable[[Population], float],
		cost_individual_func: Callable[[Individual], float],
		get_best_individual: Callable[[Population], Individual],
		get_init_population: Callable[[int], Population],
		selection_func: Callable[[Population], Population],
		crossing_func: Callable[[Population, float], Population],
		mutation_func: Callable[[Population, float, float], Population],
		succession_func: Callable[[Population], Population] = None,
		crossing_probability: float = 0.9,
		mutation_probability: float = 0.2,
		mutation_power: float = 5,
		number_individual: int = 100,
		max_iteration: int = 100

) -> Tuple[Population, Population, Population, Individual]:
	populations: List[Population] = []
	init_population = get_init_population(number_individual)
	best_population_cost = init_population_cost = cost_population_func(init_population)
	populations.append(init_population)
	best_individual = get_best_individual(init_population)
	best_individual_cost = cost_individual_func(best_individual)
	curr_population = init_population
	best_population = init_population
	pop_with_best_point = []
	##########################
	for _ in range(max_iteration):
		curr_population = selection_func(curr_population)
		curr_population = crossing_func(curr_population, crossing_probability)
		curr_population = mutation_func(curr_population, mutation_probability, mutation_power)
		curr_population_cost = cost_population_func(curr_population)
		if best_population_cost > curr_population_cost:
			best_population = curr_population
		curr_best_individual = get_best_individual(curr_population)
		curr_best_individual_cost = cost_individual_func(curr_best_individual)

		if curr_best_individual_cost < best_individual_cost:
			best_individual = curr_best_individual
			best_individual_cost = curr_best_individual_cost
			pop_with_best_point = curr_population
		if succession_func is not None:
			curr_population = succession_func(curr_population)
		populations.append(curr_population)
		# print(curr_best_individual_cost)
	# print('--------')
	# print(get_cost(get_best_individual(init_population)))
	# print(get_cost(best_individual))
	# for population in populations:
		
	# 	curr_best_individual = get_best_individual(population)
	# 	curr_best_individual_cost = get_cost(curr_best_individual)
	# 	print(curr_best_individual_cost)
	# for individual in init_population:
	# 	print(individual)
	# print('--------')
	# print(best_individual)
	# print(best_individual_cost)
	return (init_population, best_population, pop_with_best_point, best_individual)


# for _ in range(20):
if True:
	init_pop, best_pop, pop_with_best_point, best_individual =  run_simulation(
		cost_population_func=get_cost_population,
		cost_individual_func=get_cost,
		get_best_individual=get_best_individual,
		get_init_population=get_random_population,
		selection_func=tournament_selection,
		crossing_func=single_point_crossing_population,
		mutation_func=gaussian_mutation_population,
		succession_func=None,
		crossing_probability=0.8,
		mutation_probability=0.2,
		mutation_power=1.8,
		number_individual=100,
		max_iteration=300
	)
	pop_f1_init_x1 = []
	pop_f1_init_y1 = []

	pop_f1_best_x1 = []
	pop_f1_best_y1 = []

	pop_f1_with_best_x1 = []
	pop_f1_with_best_y1 = []
	for ind in init_pop:
		pop_f1_init_x1.append(ind.genome[0])
		pop_f1_init_y1.append(ind.genome[1])

	for ind in best_pop:	
		pop_f1_best_x1.append(ind.genome[0])
		pop_f1_best_y1.append(ind.genome[1])


	for ind in pop_with_best_point:	
		pop_f1_with_best_x1.append(ind.genome[0])
		pop_f1_with_best_y1.append(ind.genome[1])

	plot_contour_with_two_populations(get_f1, title_f1, pop_f1_init_x1, pop_f1_init_y1, pop_f1_best_x1, pop_f1_best_y1, pop_f1_with_best_x1, pop_f1_with_best_y1)
	print(get_cost(best_individual))
	plt.show()
	# print(str(best_ind))
	# print(best_cost)
	# print('----------')


"""
pop = get_random_population()
# pop =  sorted(pop, key=get_cost)
# for ind in pop:
# 	print(str(ind) + ' cost: '+ str(get_cost(ind)))


new_pop = tournament_selection(pop)
# print('--------')
new_pop = sorted(new_pop, key=get_cost)
ind_1 = new_pop[0]
ind_2 = new_pop[6]
# print('before crossing: ')
# print(ind_1)
# print(ind_2)
# print('after crossing: ')
# new_ind_1, new_ind_2 = single_point_crossing(ind_1, ind_2)
# print(new_ind_1)
# print(new_ind_2)
# for ind in new_pop:
# 	print(str(ind) + ' cost: '+ str(get_cost(ind)))
gaussian_mutation(ind_1, 2)


# plot_surface(get_f1, title_f1)
# plot_surface(get_f2, title_f2)

# plot_contour(get_f1, title_f1)
# plot_contour(get_f2, title_f2)

plt.show()
"""

# pop = get_random_population(5)
# pop =  sorted(pop, key=get_cost)
# for ind in pop:
# 	print(str(ind) + ' cost: '+ str(get_cost(ind)))

# print('----------')
# new_pop = gaussian_mutation_population(pop, 0.4, 0.5)
# new_pop =  sorted(new_pop, key=get_cost)
# for ind in new_pop:
# 	print(str(ind) + ' cost: '+ str(get_cost(ind)))


# plot_contour(get_f2, title_f1)
# plt.show()