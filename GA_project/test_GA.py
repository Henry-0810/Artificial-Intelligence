import pandas as pd
import numpy as np
import yfinance as yf
from google.colab import drive

# start_date = "2020-02-11"
start_date = "2014-02-11"
end_date = "2025-02-11"

stocks = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA"]

# Added 5 more stocks that are not tech related to increase diversity, stocks 1st version result in poor Sharpe Ratio due to same type/industry of stocks
stocks_2nd_version = [
    "AAPL",
    "TSLA",
    "MSFT",
    "AMZN",
    "NVDA",
    "JNJ",
    "XOM",
    "KO",
    "WMT",
    "V",
]

data = yf.download(stocks, start=start_date, end=end_date, interval="1d")
data_10 = yf.download(stocks_2nd_version, start=start_date, end=end_date, interval="1d")
print("\n", data.columns)

close_data = data["Close"]
close_data_10 = data_10["Close"]
drive.mount("/content/drive")

close_data.to_csv("/content/drive/My Drive/stock_data.csv")
close_data_10.to_csv("/content/drive/My Drive/stock_data_10.csv")

stock_data = pd.read_csv(
    "/content/drive/My Drive/stock_data.csv", index_col=0, parse_dates=True
)
stock_data.head(10)

daily_returns = stock_data.pct_change().dropna()

# multiplying 252 because 1 year have 252 working days
expected_returns = daily_returns.mean() * 252

covariance_matrix = daily_returns.cov() * 252

print("Daily Return:\n", daily_returns)
print("\nExpected Return:\n", expected_returns)
print("\nCovariance Matrix:\n", covariance_matrix)


def sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate=0.0):
    """
    Parameters:
    - weights: Array of portfolio allocation (sum to 1)
    - expected_returns: Expected return of each stocks
    - covariance_matrix: Covariance matrix of stock returns
    - risk_free_rates: default = 0

    Returns:
    - Sharpe Ratio (higher = better/optimized)
    """

    # Reference to Sharpe Ratio formulas above
    portfolio_return = np.dot(weights, expected_returns)

    # Transpose weights to get a single variance value
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix * 252, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)

    return (portfolio_return - risk_free_rate) / portfolio_stddev


class Problem:
    def __init__(self) -> None:
        self.number_of_genes = 5  # 10 for GA Loop 3
        self.min_value = 0
        self.max_value = 1
        self.fitness_function = sharpe_ratio
        self.acceptable_fitness = 1


from copy import deepcopy


class PortfolioIndividual:
    def __init__(self, prob):
        """
        Initializes an individual portfolio (chromosomes)

        Parameters:
        - prob: An Instance of the Problem class
        """
        self.chromosomes = np.random.dirichlet(np.ones(prob.number_of_genes), size=1)[0]
        self.fitness_function = prob.fitness_function
        self.fitness = self.fitness_function(
            self.chromosomes, expected_returns, covariance_matrix
        )

    def crossover(self, parent2, explore_crossover=0.1):
        """
        Linear Interpolation Crossover, more details above

        Parameters:
        - parent2: Another PortfolioIndividual to crossover with
        - explore_crossover: Exploration factor (default 0.1)

        Returns:
        - 2 children PortfolioIndividuals
        """
        alpha = np.random.uniform(-explore_crossover, 1 + explore_crossover)

        child1 = deepcopy(self)
        child2 = deepcopy(parent2)

        child1.chromosomes = (
            alpha * self.chromosomes + (1 - alpha) * parent2.chromosomes
        )
        child2.chromosomes = (
            alpha * parent2.chromosomes + (1 - alpha) * self.chromosomes
        )

        return child1, child2

    def mutate(self, mutation_rate=0.1, mutation_strength=0.05):
        """
        Applies mutation by slightly adding noise to random portfolio weights

        Parameters:
        - mutation_rate: Probability of mutation to happen for each weight
        - mutation_strength: Standard deviation of noise added

        Returns:
        - Mutated PortfolioIndividual
        """
        mutated_portfolio = deepcopy(self)

        for i in range(len(mutated_portfolio.chromosomes)):
            if np.random.rand() < mutation_rate:
                mutated_portfolio.chromosomes[i] += np.random.normal(
                    0, mutation_strength
                )

        mutated_portfolio.chromosomes = np.clip(mutated_portfolio.chromosomes, 0, 1)

        mutated_portfolio.chromosomes /= np.sum(mutated_portfolio.chromosomes)

        return mutated_portfolio


class Parameters:
    def __init__(self):
        self.population_size = 50
        self.number_of_generations = 100
        self.mutation_rate = 0.1
        self.mutation_strength = 0.05
        self.explore_crossover_range = 0.2
        self.birth_rate_per_generation = 1
        self.tournament_size = 3


def choose_parents(population, tournament_size=3):
    """
    Tournament Selection: choose 3 random parents, calculate their sharpe ratio and return the highest.

    Parameters:
    - tournament_size: number of contestents (i.e. number of parents chosen to compete in this case)

    Returns:
    - 2 parents from 2 tournaments with highest fitness function (Sharpe Ratio)
    """
    tournament1 = np.random.choice(population, tournament_size)
    tournament2 = np.random.choice(population, tournament_size)

    parent1 = max(tournament1, key=lambda x: x.fitness)
    parent2 = max(tournament2, key=lambda x: x.fitness)

    return parent1, parent2


def run_genetic(prob, params, experiment_name="Experiment 1"):
    # 1. Read variables
    num_of_population = params.population_size
    rate_of_gene_mutation = params.mutation_rate
    mutation_strength = params.mutation_strength
    explore_crossover = params.explore_crossover_range
    max_num_of_generations = params.number_of_generations

    fitness_function = prob.fitness_function
    acceptable_fitness = prob.acceptable_fitness
    num_of_child_per_generation = num_of_population * params.birth_rate_per_generation

    # 2. Create population
    population = []
    best_solution = PortfolioIndividual(prob)
    best_solution.fitness = -np.inf

    for i in range(num_of_population):
        new_individual = PortfolioIndividual(prob)
        new_individual.fitness = fitness_function(
            new_individual.chromosomes, expected_returns, covariance_matrix
        )

        if new_individual.fitness > best_solution.fitness:
            best_solution = deepcopy(new_individual)

        population.append(new_individual)

    print(f"Initial Population Size: {len(population)}")

    # 3. Start GA Evolution Loop
    for gen in range(max_num_of_generations):
        children = []

        # Generate children through crossover and mutation
        while len(children) < num_of_child_per_generation:
            parent1, parent2 = choose_parents(
                population=population, tournament_size=params.tournament_size
            )

            child1, child2 = parent1.crossover(parent2, explore_crossover)

            child1 = child1.mutate(rate_of_gene_mutation, mutation_strength)
            child2 = child2.mutate(rate_of_gene_mutation, mutation_strength)

            child1.fitness = fitness_function(
                child1.chromosomes, expected_returns, covariance_matrix
            )
            child2.fitness = fitness_function(
                child2.chromosomes, expected_returns, covariance_matrix
            )

            children.append(child1)
            children.append(child2)

        # 4. Add children to new population
        population += children

        # 5. Sort
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        # 6. Cull population to fixed size
        population = population[:num_of_population]

        if population[0].fitness > best_solution.fitness:
            best_solution = deepcopy(population[0])
            print(
                f"Generation {gen + 1}: New Best Sharpe Ratio = {best_solution.fitness}"
            )

        if best_solution.fitness > acceptable_fitness:
            print("Optimal portfolio found, stopping early.")
            break

    return population, best_solution
