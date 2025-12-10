import random
from typing import List
from .genome import Genome

class Population:
    def __init__(self, size: int = 10):
        """
        Initialize a population of random genomes.
        
        Args:
            size: The number of individuals in the population.
        """
        self.size = size
        self.generation = 0
        self.individuals: List[Genome] = [Genome() for _ in range(size)]

    def evolve(self, elitism_count: int = 2, mutation_rate: float = 0.1):
        """
        Evolve the population to the next generation.
        
        Args:
            elitism_count: Number of top individuals to carry over unchanged.
            mutation_rate: Probability of mutation for offspring.
        """
        # Sort by fitness (descending)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        next_gen = []
        
        # Elitism
        next_gen.extend([g.copy() for g in self.individuals[:elitism_count]])
        
        # Generate offspring
        while len(next_gen) < self.size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            offspring = self.crossover(parent1, parent2)
            offspring.mutate(mutation_rate)
            
            next_gen.append(offspring)
            
        self.individuals = next_gen
        self.generation += 1

    def tournament_selection(self, k: int = 3) -> Genome:
        """
        Selects the best individual from k random individuals.
        """
        candidates = random.sample(self.individuals, k)
        return max(candidates, key=lambda x: x.fitness)

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """
        Performs Variable Split Crossover between two parents.
        Allows offspring to have different length than parents (depth evolution).
        """
        genes1 = parent1.genes
        genes2 = parent2.genes
        
        # Variable Split:
        # Take 0..split1 from Parent 1
        # Take split2..end from Parent 2
        # This allows the network to grow or shrink
        
        # Ensure we don't split in the middle of a critical block if possible, 
        # but for now if a split creates a broken or useless connection, that child will just get a low fitness and die out. If it creates a weird but working connection, it might be a breakthrough.
        
        split1 = random.randint(0, len(genes1))
        split2 = random.randint(0, len(genes2))
        
        new_genes = [g.copy() for g in genes1[:split1]] + [g.copy() for g in genes2[split2:]]
        
        # Sanity Check 1: Ensure Flatten and Linear exist at the end
        # We remove any existing Flatten/Linear in the middle to avoid early termination
        new_genes = [g for g in new_genes if g['type'] not in ['Flatten', 'Linear']]
        
        # Always append the head
        new_genes.append({'type': 'Flatten'})
        new_genes.append({'type': 'Linear', 'out_features': 10})
        
        # Sanity Check 2: Ensure at least one Conv layer exists before Flatten
        has_conv = any(g['type'] == 'Conv2d' for g in new_genes)
        if not has_conv:
            # If we accidentally removed all conv layers, insert a default one
            new_genes.insert(0, {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1})
            new_genes.insert(1, {'type': 'ReLU'})
            new_genes.insert(2, {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2})
            
        return Genome(genes=new_genes)
