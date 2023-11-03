import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import copy
import random
import numpy as np
from plot import (
    plot_surface,
    title_f1,
    title_f2,
    plot_contour,
    plot_contour_with_populations_part,
    plot_bar_chart_with_populations,
)
from functions import DOMAIN, get_f1, get_f2

Genome = List[int]

GENOME_LENGTH = 4


class Individual:
    # genome[0] = x1
    # genome[1] = y1
    # genome[2] = x2
    # genome[3] = y2
    def __init__(self, genome: Genome):
        self.genome = genome

    def __str__(self):
        string_genome = [str(round(gene, 5)) for gene in self.genome]
        return "[" + ", ".join(string_genome) + "]"


Population = List[Individual]


class PopulationPartToPlot:
    def __init__(self, data: List[int], title: str, color: str) -> None:
        self.data = data
        self.title = title
        self.color = color
        self.data_x = [data_x[0] for data_x in data]
        self.data_y = [data_y[1] for data_y in data]


def population_for_plot(population: Population, number_function: int) -> List[int]:
    """
    number_function denotes which part of genome of individual should be returned
    """
    individuals_one_function = []
    if number_function == 1:
        individuals_one_function = [individual.genome[:2] for individual in population]
    elif number_function == 2:
        individuals_one_function = [individual.genome[2:] for individual in population]
    else:
        raise ValueError("Number_function can be either 1 or 2!")
    return individuals_one_function


def get_cost(individual: Individual) -> float:
    return get_f1(individual.genome[0], individual.genome[1]) + get_f2(
        individual.genome[2], individual.genome[3]
    )


def get_cost_population(population: Population) -> float:
    return sum([get_cost(genome) for genome in population])


def sort_population(population: Population) -> Population:
    return sorted(population, key=get_cost)


def get_best_individual(population: Population) -> Individual:
    sorted_pop = sort_population(population)
    return sorted_pop[0]


def get_random_population(number_individuals=20, domain=DOMAIN) -> Population:
    """
    returns a population considering constraints of domain
    """
    population: Population = []
    for _ in range(number_individuals):
        genome = get_random_genome(domain)
        individual = Individual(genome)
        population.append(individual)
    return population


def get_random_genome(domain=DOMAIN) -> Genome:
    """
    returns a genome considering constraints of domain
    """
    genome: Genome = []
    for i in range(GENOME_LENGTH):
        if i % 2 == 0:
            genome.append(random.uniform(DOMAIN.x_lower_bound, DOMAIN.x_upper_bound))
        else:
            genome.append(random.uniform(DOMAIN.y_lower_bound, DOMAIN.y_upper_bound))
    return genome


def get_random_genome_gaussian(
    standard_deviation_x1: float = 1.5,
    standard_deviation_y1: float = 1.5,
    mean_x1: float = -0.3,
    mean_y1: float = -0.9,
    domain=DOMAIN,
) -> Genome:
    """
    returns a genome from gaussian distribution
    """
    genome: Genome = []
    for i in range(GENOME_LENGTH):
        if i == 0:
            genome.append(np.random.normal(mean_x1, standard_deviation_x1))
        elif i == 1:
            genome.append(np.random.normal(mean_y1, standard_deviation_y1))
        elif i == 2:
            genome.append(random.uniform(DOMAIN.x_lower_bound, DOMAIN.x_upper_bound))
        else:
            genome.append(random.uniform(DOMAIN.y_lower_bound, DOMAIN.y_upper_bound))
    return genome


def get_random_gaussian_population(
    number_individuals=20,
    standard_deviation_x1: float = 1.5,
    standard_deviation_y1: float = 1.5,
    mean_x1: float = -0.3,
    mean_y1: float = -0.9,
    domain=DOMAIN,
) -> Population:
    """
    returns a population considering constraints of domain and gaussian distribution
    """
    population: Population = []
    for _ in range(number_individuals):
        genome = get_random_genome_gaussian(
            standard_deviation_x1, standard_deviation_y1, mean_x1, mean_y1, domain
        )
        individual = Individual(genome)
        population.append(individual)
    return population


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


def single_point_crossing(
    individual_1: Individual, individual_2: Individual
) -> Tuple[Individual, Individual]:
    if len(individual_1.genome) != len(individual_2.genome):
        raise ValueError("genomes should have the same length!")

    point_index = random.randint(1, len(individual_1.genome) - 1)

    new_genome_1 = (
        individual_1.genome[0:point_index] + individual_2.genome[point_index:]
    )
    new_genome_2 = (
        individual_2.genome[0:point_index] + individual_1.genome[point_index:]
    )

    return Individual(new_genome_1), Individual(new_genome_2)


def single_point_crossing_population(
    population: Population, crossing_probability: float
) -> Population:
    """
    crossing_probability should have value between [0, 1]
    """
    current_population_copy = [copy.deepcopy(individual) for individual in population]
    new_population: Population = []
    while len(current_population_copy) > 1:
        rand_number = random.uniform(0, 1)
        index_parent_1 = random.randint(0, len(current_population_copy) - 1)
        parent_1 = current_population_copy[index_parent_1]
        current_population_copy.pop(index_parent_1)
        index_parent_2 = random.randint(0, len(current_population_copy) - 1)
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


def gaussian_mutation(
    individual: Individual, mutation_probability: float, mutation_power: float
):
    for i in range(len(individual.genome)):
        rand_number = random.uniform(0, 1)
        if mutation_probability == 1 or rand_number < mutation_probability:
            mutation_offset = mutation_power * np.random.normal(0, 1)
            individual.genome[i] += mutation_offset


def gaussian_mutation_population(
    population: Population, mutation_probability: float, mutation_power: float
) -> Population:
    """
    mutation_probability should have value between [0, 1]
    """
    new_population = [copy.deepcopy(individual) for individual in population]
    for individual in new_population:
        gaussian_mutation(individual, mutation_probability, mutation_power)

    return new_population


"""
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
"""


# def get_population_with_best_individual(populations: List[Population], get_cost: Callable[[Individual], float] = get_cost) -> Population:
#     best_population: Population = []
#     best_individual_cost = None
#     a = 1
# 	for population in populations:
#         for individual in population:
#             current_individual_cost = get_cost(individual)
#             if best_individual_cost is None or current_individual_cost < best_individual_cost:
#                 best_individual_cost = current_individual_cost
#                 best_population = population


def get_population_with_best_individual(
    populations: List[Population], get_cost: Callable[[Individual], float] = get_cost
) -> Population:
    best_population: Population = []
    best_individual_cost = None
    for population in populations:
        for individual in population:
            current_individual_cost = get_cost(individual)
            if (
                best_individual_cost is None
                or current_individual_cost < best_individual_cost
            ):
                best_individual_cost = current_individual_cost
                best_population = population

    return best_population


def get_population_with_best_cost(
    populations: List[Population],
    get_cost_population: Callable[[Population], float] = get_cost_population,
) -> Population:
    best_population: Population = []
    best_population_cost = None
    for population in populations:
        current_population_average_cost = get_cost_population(population)
        if (
            best_population_cost is None
            or best_population_cost > current_population_average_cost
        ):
            best_population_cost = current_population_average_cost
            best_population = population

    return best_population


def run_simulation(
    # cost_population_func: Callable[[Population], float],
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
    max_iteration: int = 100,
) -> Tuple[List[Population], Individual, int]:
    populations: List[Population] = []
    init_population = get_init_population(number_individual)
    populations.append(init_population)
    best_individual = get_best_individual(init_population)
    best_individual_cost = cost_individual_func(best_individual)
    curr_population = init_population
    ##########################
    for _ in range(max_iteration):
        curr_population = selection_func(curr_population)
        curr_population = crossing_func(curr_population, crossing_probability)
        curr_population = mutation_func(
            curr_population, mutation_probability, mutation_power
        )
        curr_best_individual = get_best_individual(curr_population)
        curr_best_individual_cost = cost_individual_func(curr_best_individual)

        if curr_best_individual_cost < best_individual_cost:
            best_individual = curr_best_individual
            best_individual_cost = curr_best_individual_cost
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
    return (populations, best_individual, best_individual_cost)


def filter_population(population: Population, domain=DOMAIN):
    """
    Filters population based on constrains defined in the domain
    """
    filtered_population: Population = []
    for individual in population:
        out_range = False
        for i in range(len(individual.genome)):
            gene = individual.genome[i]
            if i % 2 == 0 and (
                gene > domain.x_upper_bound or gene < domain.x_lower_bound
            ):
                out_range = True
                break
            elif i % 2 == 1 and (
                gene > domain.y_upper_bound or gene < domain.y_lower_bound
            ):
                out_range = True
                break
        if not out_range:
            filtered_population.append(individual)
    return filtered_population


populations, best_individual, best_individual_cost = run_simulation(
    cost_individual_func=get_cost,
    get_best_individual=get_best_individual,
    get_init_population=get_random_gaussian_population,
    selection_func=tournament_selection,
    crossing_func=single_point_crossing_population,
    mutation_func=gaussian_mutation_population,
    succession_func=None,
    crossing_probability=0.8,
    mutation_probability=0.2,
    mutation_power=1.8,
    number_individual=100,
    max_iteration=100,
)
################
pop_best_avg = get_population_with_best_cost(populations)
pop_best_ind = get_population_with_best_individual(populations)
init_pop = populations[0]
# plot_contour_with_populations_part(
#     get_f1,
#     title_f1,
#     [
#         PopulationPartToPlot(
#             population_for_plot(filter_population(init_pop), 1),
#             f"init pop, avg = {round(get_cost_population(init_pop) / len(init_pop), 3)}, best = {round(get_cost(get_best_individual(init_pop)), 5)}",
#             "orange",
#         ),
#         PopulationPartToPlot(
#             population_for_plot(filter_population(pop_best_avg), 1),
#             f"pop with best avg, avg = {round(get_cost_population(pop_best_avg) / len(pop_best_avg), 3)}, best = {round(get_cost(get_best_individual(pop_best_ind)), 5)}",
#             "red",
#         ),
#         PopulationPartToPlot(
#             population_for_plot(filter_population(pop_best_ind), 1),
#             f"pop with best individual, avg = {round(get_cost_population(pop_best_ind) / len(pop_best_ind), 3)}, best = {round(get_cost(get_best_individual(pop_best_ind)), 5)}",
#             "green",
#         ),
#     ],
# )
####################
#
# plot_contour_with_populations_part(
#     get_f2,
#     title_f2,
#     [
#         PopulationPartToPlot(
#             population_for_plot(filter_population(init_pop), 2),
#             f"init pop, avg = {round(get_cost_population(init_pop) / len(init_pop), 3)}, best = {round(get_cost(get_best_individual(init_pop)), 5)}",
#             "orange",
#         ),
#         PopulationPartToPlot(
#             population_for_plot(filter_population(pop_best_avg), 2),
#             f"pop with best avg, avg = {round(get_cost_population(pop_best_avg) / len(pop_best_avg), 3)}, best = {round(get_cost(get_best_individual(pop_best_ind)), 5)}",
#             "red",
#         ),
#         PopulationPartToPlot(
#             population_for_plot(filter_population(pop_best_ind), 2),
#             f"pop with best individual, avg = {round(get_cost_population(pop_best_ind) / len(pop_best_ind), 3)}, best = {round(get_cost(get_best_individual(pop_best_ind)), 5)}",
#             "green",
#         ),
#     ],
# )
# plt.show()
###########################
# bar chart average #####
# n = 10
# cut_populations = populations[::n]
# populations_to_chart = []
# for pop in cut_populations:
#     populations_to_chart.append(get_cost_population(pop) / len(pop))

# generations_names = ["Gen " + str(i) for i in range(0, 101, n)]

# plot_bar_chart_with_populations(populations_to_chart, generations_names)
###
# bar chart best
n = 10
cut_populations = populations[::n]
best_ind_costs = []
for pop in cut_populations:
    best_ind_costs.append(get_cost(get_best_individual(pop)))

generations_names = ["Gen " + str(i) for i in range(0, 101, n)]

plot_bar_chart_with_populations(
    best_ind_costs,
    generations_names,
    "Best cost for individual per generation",
    "Cost for individual",
)

plt.show()
####################
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
