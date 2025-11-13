import os
import time
import random


def print_human_readable_cnf(cnf_file):
    clauses = []
    num_vars = 0
    num_clauses = 0
    with open(cnf_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            elif line.startswith('c'):
                continue
            elif line.startswith('%'):
                break
            else:
                clause = list(map(int, line.split()[:-1]))
                clauses.append(clause)
    clause_strings = []
    for i, clause in enumerate(clauses, 1):
        literals = []
        for lit in clause:
            var = abs(lit)
            sign = "" if lit > 0 else "¬"
            literals.append(f"{sign}x{var}")
        clause_str = " ∨ ".join(literals)
        clause_strings.append(f"({clause_str})")
    formula_str = " ∧ ".join(clause_strings)
    print(f"\nПолная формула:\n{formula_str}")

def fitness(individual, clauses):
    satisfied = 0
    for clause in clauses:
        for lit in clause:
            var = abs(lit)
            if (lit > 0 and individual[var-1]) or (lit < 0 and not individual[var-1]):
                satisfied += 1
                break
    return satisfied

def initialize_population(pop_size, num_vars):
    return [[random.choice([True, False]) for _ in range(num_vars)] for _ in range(pop_size)]

def select_parents(population, fitnesses, num_parents):
    parents = []
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    elite_size = int(0.7 * num_parents) # Квота 30% для не лучших - что бы сохранить генетическое разнообразие
    for i in range(elite_size):
        parents.append(sorted_population[i][0])
    for _ in range(num_parents - elite_size):
        random_index = random.randint(0, len(population) - 1)
        parents.append(population[random_index])
    return parents

def crossover(parent1, parent2):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = not individual[i]
    return individual

def genetic_algorithm_sat(cnf_file, pop_size=20, max_pops=2300, max_generations=11, mutation_rate=0.03, civilizations=60):
    clauses = []
    num_vars = 0
    num_clauses = 0
    with open(cnf_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            elif line.startswith('c'):
                continue
            elif line.startswith('%'):
                break
            else:
                clause = list(map(int, line.split()[:-1]))
                clauses.append(clause)
    start_time = time.time()
    best_individual = None
    best_civilization = -1
    best_generation = -1
    best_pop_size = -1
    for civilization in range(civilizations):
        current_pop_size = int(pop_size + (max_pops - pop_size) * (civilization + 1) / civilizations)
        population = initialize_population(current_pop_size, num_vars)
        best_fitness = 0
        for generation in range(max_generations):
            fitnesses = [fitness(ind, clauses) for ind in population]
            current_best_fitness = max(fitnesses)
            current_best_index = fitnesses.index(current_best_fitness)
            current_best_individual = population[current_best_index]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                best_civilization = civilization
                best_generation = generation
                best_pop_size = current_pop_size
                if best_fitness == num_clauses:
                    return True, time.time() - start_time, num_vars, num_clauses, best_civilization, best_generation, best_pop_size
            parents = select_parents(population, fitnesses, current_pop_size)
            elite_size = int(0.1 * current_pop_size)
            elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:elite_size]
            elites = [ind for ind, fit in elites]
            next_generation = elites.copy()
            for i in range(0, current_pop_size - elite_size, 2):
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                next_generation.extend([child1, child2])
            if len(next_generation) < current_pop_size:
                random_individuals = initialize_population(current_pop_size - len(next_generation), num_vars)
                next_generation.extend(random_individuals)
            population = next_generation
    return False, time.time() - start_time, num_vars, num_clauses, best_civilization, best_generation, best_pop_size



def main():
    stat_files_red = 0

    dataset_dir = './datasets'
    total_files = 0
    satisfiable_files = 0
    total_vars = 0
    total_clauses = 0
    total_time = 0
    total_civilizations = 0
    total_generations = 0
    total_pop_sizes = 0
    min_time = float('inf')
    max_time = 0
    min_civilizations = float('inf')
    max_civilizations = 0
    min_generations = float('inf')
    max_generations = 0
    min_pop_size = float('inf')
    max_pop_size = 0
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.cnf'):
            stat_files_red += 1
            filepath = os.path.join(dataset_dir, filename)
            print(f"Решаю {filename}...")
            is_sat, time_taken, num_vars, num_clauses, best_civilization, best_generation, best_pop_size = genetic_algorithm_sat(filepath)
            print(f"Время: {time_taken:.4f} секунд\n")
            total_files += 1
            total_vars += num_vars
            total_clauses += num_clauses
            total_time += time_taken
            if is_sat:
                satisfiable_files += 1
            total_civilizations += best_civilization
            total_generations += best_generation
            total_pop_sizes += best_pop_size
            if time_taken < min_time:
                min_time = time_taken
            if time_taken > max_time:
                max_time = time_taken
            if best_civilization < min_civilizations:
                min_civilizations = best_civilization
            if best_civilization > max_civilizations:
                max_civilizations = best_civilization
            if best_generation < min_generations:
                min_generations = best_generation
            if best_generation > max_generations:
                max_generations = best_generation
            if best_pop_size < min_pop_size:
                min_pop_size = best_pop_size
            if best_pop_size > max_pop_size:
                max_pop_size = best_pop_size
    if total_files > 0:
        satisfiable_percentage = (satisfiable_files / total_files) * 100
        avg_vars = total_vars / total_files
        avg_clauses = total_clauses / total_files
        avg_civilizations = total_civilizations / total_files
        avg_generations = total_generations / total_files
        avg_pop_sizes = total_pop_sizes / total_files
        avg_time = total_time / total_files
    else:
        satisfiable_percentage = 0
        avg_vars = 0
        avg_clauses = 0
        avg_civilizations = 0
        avg_generations = 0
        avg_pop_sizes = 0
        avg_time = 0

    print(f"Файлов в датасете: {stat_files_red}")
    print(f"Процент выполнимых формул: {satisfiable_percentage:.2f}%")
    print(f"Среднее количество переменных: {avg_vars:.2f}")
    print(f"Среднее количество клауз: {avg_clauses:.2f}")
    print(f"Суммарное время работы: {total_time:.4f} секунд")
    print(f"Среднее количество цивилизаций: {avg_civilizations:.2f}")
    print(f"Среднее количество поколений: {avg_generations:.2f}")
    print(f"Среднее количество индивидов в популяции: {avg_pop_sizes:.2f}")
    print(f"Минимальное время: {min_time:.4f} секунд")
    print(f"Максимальное время: {max_time:.4f} секунд")
    print(f"Минимальное количество цивилизаций: {min_civilizations}")
    print(f"Максимальное количество цивилизаций: {max_civilizations}")
    print(f"Минимальное количество поколений: {min_generations}")
    print(f"Максимальное количество поколений: {max_generations}")
    print(f"Минимальное количество индивидов в популяции: {min_pop_size}")
    print(f"Максимальное количество индивидов в популяции: {max_pop_size}")
    print(f"\n\nСреднее время работы: {avg_time:.4f} секунд")

if __name__ == "__main__":
    main()