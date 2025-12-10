import torch
from neuroevolve.population import Population
from neuroevolve.utils import get_device
from neuroevolve.trainer import get_data_loaders, train_model
import math

def main():
    print("NeuroEvolve: Initializing...")
    
    # Configuration
    POP_SIZE = 5
    GENERATIONS = 100
    EPOCHS_PER_GENOME = 10 # Increased for better convergence
    BATCH_SIZE = 256 # Match the Actor batch size for consistent results
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Data
    print(f"Loading CIFAR-10 data (Batch Size: {BATCH_SIZE})...")
    train_loader, val_loader = get_data_loaders(batch_size=BATCH_SIZE)
    
    # Initialize Population
    pop = Population(size=POP_SIZE)
    print(f"Initialized population of size {pop.size}")
    
    # Initialize Ray
    import ray
    # Check if already initialized (for notebooks/interactive)
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}}) # Help with Windows
        
    # Create Actor Pool
    num_actors = 5
    print(f"Creating {num_actors} Ray Actors...")
    from neuroevolve.trainer import TrainActor
    actors = [TrainActor.remote(batch_size=256) for _ in range(num_actors)]
    
    # History Tracking
    history = []
    import pandas as pd
    
    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen} ---")
        
        # Distribute Training
        futures = []
        for i, genome in enumerate(pop.individuals):
            actor = actors[i % num_actors] # Round-robin assignment
            print(f"Dispatching Genome {i} (LR={genome.learning_rate})...")
            
            # Call remote method
            future = actor.train.remote(genome, epochs=EPOCHS_PER_GENOME)
            futures.append(future)
            
        # Wait for all results
        print("Waiting for training results...")
        results = []
        for i, future in enumerate(futures):
            try:
                results.append(ray.get(future))
            except ray.exceptions.ActorUnavailableError:
                print(f"Actor crashed for Genome {i}. Assigning fitness 0.")
                results.append((0.0, 0)) # Dummy result
            except Exception as e:
                print(f"Error training Genome {i}: {e}")
                results.append((0.0, 0))
        
        # Process Results
        gen_fitnesses = []
        for i, (val_acc, param_count) in enumerate(results):
            genome = pop.individuals[i]
            
            if param_count > 0:
                penalty = 0.005 * math.log(param_count)
            else:
                penalty = 0
                
            genome.fitness = val_acc - penalty
            gen_fitnesses.append(genome.fitness)
            print(f"  Genome {i} -> Val Acc: {val_acc:.4f}, Params: {param_count}, Fitness: {genome.fitness:.4f}")
            
        # Log History
        best_fit = max(gen_fitnesses)
        avg_fit = sum(gen_fitnesses) / len(gen_fitnesses)
        history.append({
            'generation': gen,
            'best_fitness': best_fit,
            'avg_fitness': avg_fit
        })
        
        # Save to CSV immediately so dashboard can update
        pd.DataFrame(history).to_csv('evolution_history.csv', index=False)
        
        # Evolve
        if gen < GENERATIONS - 1:
            print("Evolving...")
            pop.evolve()
            
    # Best Model
    best_genome = max(pop.individuals, key=lambda x: x.fitness)
    print(f"\nBest Genome Fitness: {best_genome.fitness:.4f}")
    print(f"Best Genome LR: {best_genome.learning_rate}")
    print("Best Architecture Genes:")
    for gene in best_genome.genes:
        print(gene)
        
    # Verify decoding of best model
    print("\nDecoding Best Model...")
    try:
        model = best_genome.decode()
        print(model)
        print("Success!")
        
        # Save Best Model
        print("\nSaving Best Model...")
        import json
        
        # Save Genome
        with open('best_genome.json', 'w') as f:
            json.dump(best_genome.genes, f, indent=4)
        print("Saved best_genome.json")
        
        # Save Weights
        print("Retraining best model to save weights...")
        
        # We use return_model=True to get the actual trained instance
        _, _, trained_model = train_model(
            best_genome, 
            train_loader, 
            val_loader, 
            epochs=50, 
            device=device, 
            verbose=True,
            return_model=True
        )
        
        torch.save(trained_model.state_dict(), 'best_model.pth')
        print("Saved best_model.pth")
        
    except Exception as e:
        print(f"Error saving best model: {e}")

if __name__ == "__main__":
    main()
