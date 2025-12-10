# NeuroEvolve: Technical Implementation Analysis

## Analysis Overview

NeuroEvolve is an intelligent agent system that uses genetic algorithms to automatically design and optimize neural network architectures for CIFAR-10 image classification. The system encodes neural networks as genomes, evolves them over generations using selection, crossover, and mutation operators, and evaluates fitness through distributed training on GPU.

---

## Entry Points

| File | Location | Purpose |
|------|----------|---------|
| `main.py:7` | `main()` function | Primary entry point for evolutionary optimization |
| `dashboard.py:1-61` | Streamlit app | Real-time training visualization dashboard |

---

## Core Implementation

### 1. Genetic Encoding (`genome.py:7-259`)

The `Genome` class encodes neural network architectures as a list of layer dictionaries.

#### 1.1 Genome Initialization (`genome.py:8-27`)
- Accepts optional `genes` list and `learning_rate` parameters
- If no genes provided, calls `_create_random_genes()` at line 21
- Learning rate randomly selected from `[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]` at line 27
- Fitness initialized to `0.0` at line 17

#### 1.2 Random Architecture Generation (`genome.py:29-64`)
The `_create_random_genes()` method creates initial architectures:

| Step | Lines | Operation |
|------|-------|-----------|
| Depth selection | 34 | Randomly choose 1-3 convolutional blocks |
| Conv2d creation | 38-48 | `out_channels` from `[16, 32, 64, 128]`, `kernel_size` from `[3, 5]` |
| Activation | 51 | Always adds `ReLU` after convolution |
| Optional Dropout | 54-55 | 30% probability, `p` from `[0.1, 0.3, 0.5]` |
| Pooling | 58 | Always adds `MaxPool2d(kernel_size=2, stride=2)` |
| Head | 61-62 | `Flatten` + `Linear(out_features=10)` for CIFAR-10 |

#### 1.3 Genome Decoding (`genome.py:66-173`)
The `decode()` method converts genomes to `torch.nn.Module`:

**Phase 1: Layer Construction (lines 76-131)**
- Tracks `current_channels` starting from input shape (default `(3, 32, 32)` at line 66)
- Builds `feature_extractor_layers` for Conv/Pool/Activation layers (lines 85-121)
- Builds `classifier_layers` for Linear layers (lines 86, 126-130)
- Uses `is_flattened` boolean to track where to place Dropout layers (line 88)

**Phase 2: Model Assembly (lines 132-172)**
- Creates `nn.Sequential` model at line 133
- Performs dummy forward pass at lines 140-148 to compute flatten size
- Raises `ValueError` at line 148 if architecture is invalid
- Fixes `Linear` layer input features at line 158 using computed flatten size

**Supported Layer Types:**
| Type | Handler Lines | PyTorch Class |
|------|---------------|---------------|
| `Conv2d` | 93-101 | `nn.Conv2d` |
| `BatchNorm2d` | 104-105 | `nn.BatchNorm2d` |
| `ReLU` | 107-108 | `nn.ReLU` |
| `MaxPool2d` | 110-113 | `nn.MaxPool2d` |
| `Dropout` | 115-121 | `nn.Dropout` or `nn.Dropout2d` |
| `Flatten` | 123-124 | Sets `is_flattened` flag |
| `Linear` | 126-130 | `nn.Linear` (placeholder) |

#### 1.4 Mutation Operators (`genome.py:175-252`)
The `mutate()` method at line 175 applies random mutations with probability `mutation_rate`:

| Mutation Type | Method | Lines | Behavior |
|---------------|--------|-------|----------|
| `add_layer` | `_add_random_layer()` | 191-222 | Inserts Conv2d/BatchNorm2d/ReLU/Dropout before Flatten |
| `remove_layer` | `_remove_random_layer()` | 224-235 | Removes random layer before Flatten if > 1 layers exist |
| `modify_param` | `_modify_random_param()` | 237-252 | Modifies Conv2d channels/kernel or Dropout probability |
| `modify_lr` | Direct assignment | 189 | Selects new learning rate from predefined options |

#### 1.5 Genome Copy (`genome.py:254-258`)
The `copy()` method creates deep copy using `copy.deepcopy()` for genes, preserving fitness.

---

### 2. Population Management (`population.py:5-91`)

The `Population` class manages a collection of genomes and evolutionary operators.

#### 2.1 Population Initialization (`population.py:6-15`)
- Parameters: `size` (default 10)
- Attributes: `self.size`, `self.generation` (starts at 0), `self.individuals` list of `Genome` objects

#### 2.2 Evolution (`population.py:17-44`)
The `evolve()` method advances population to next generation:

| Step | Lines | Operation |
|------|-------|-----------|
| Sort by fitness | 26 | `self.individuals.sort(key=lambda x: x.fitness, reverse=True)` |
| Elitism | 31 | Copy top `elitism_count` (default 2) individuals to next generation |
| Offspring generation | 34-41 | Loop until population reaches target size |
| Parent selection | 35-36 | Two calls to `tournament_selection()` |
| Crossover | 38 | `self.crossover(parent1, parent2)` |
| Mutation | 39 | `offspring.mutate(mutation_rate)` |
| Update state | 43-44 | Replace individuals, increment generation counter |

#### 2.3 Tournament Selection (`population.py:46-51`)
Selects best individual from `k=3` random candidates based on fitness.

#### 2.4 Crossover (`population.py:53-90`)
**Variable Split Crossover** allows architecture depth to evolve:

| Step | Lines | Operation |
|------|-------|-----------|
| Get parent genes | 58-59 | Extract gene lists from both parents |
| Random splits | 69-70 | `split1` from parent1, `split2` from parent2 |
| Combine genes | 72 | `genes1[:split1] + genes2[split2:]` |
| Sanitize | 76 | Remove any `Flatten`/`Linear` from middle |
| Add head | 79-80 | Always append `Flatten` + `Linear(out_features=10)` |
| Ensure Conv exists | 83-88 | If no Conv2d, insert default conv block |

---

### 3. Training System (`trainer.py:1-145`)

#### 3.1 Data Loading (`trainer.py:9-32`)
The `get_data_loaders()` function:

| Step | Lines | Operation |
|------|-------|-----------|
| Training transforms | 13-18 | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, normalize |
| Test transforms | 20-23 | Just `ToTensor` and normalize |
| Normalization values | 17, 22 | Mean `(0.4914, 0.4822, 0.4465)`, Std `(0.2023, 0.1994, 0.2010)` |
| Dataset download | 26, 29 | CIFAR-10 to `./data` directory |

#### 3.2 Model Training (`trainer.py:34-120`)
The `train_model()` function trains decoded genomes:

**Model Decoding (lines 50-58):**
- Calls `genome.decode()` with `input_shape=(3, 32, 32)`
- Returns `(0.0, 0)` for invalid architectures

**Training Loop (lines 60-90):**
| Component | Line | Value |
|-----------|------|-------|
| Loss function | 63 | `nn.CrossEntropyLoss()` |
| Optimizer | 64 | `optim.Adam(model.parameters(), lr=genome.learning_rate)` |
| Forward pass | 78 | `outputs = model(inputs)` |
| Backpropagation | 80-81 | `loss.backward()`, `optimizer.step()` |

**Validation (lines 92-105):**
- Sets `model.eval()` at line 93
- Computes accuracy with `torch.no_grad()` context
- Returns `val_accuracy` as decimal (0.0 to 1.0)

**Cleanup (lines 112-120):**
- Deletes model and optimizer if not returning model
- Calls `torch.cuda.empty_cache()` and `gc.collect()`

#### 3.3 Distributed Training Actor (`trainer.py:126-144`)
The `TrainActor` Ray actor enables parallel genome evaluation:

| Configuration | Line | Value |
|---------------|------|-------|
| GPU allocation | 126 | `num_gpus=0.2` (fractional GPU) |
| Auto-restart | 126 | `max_restarts=-1` (infinite restarts) |
| Data loading | 134 | `num_workers=0` for Ray safety |

**Actor Methods:**
- `__init__(batch_size=256)`: Initializes device detection, loads data once per actor (lines 128-134)
- `train(genome, epochs)`: Delegates to `train_model()` function (lines 136-144)

---

### 4. Main Evolutionary Loop (`main.py:7-142`)

#### 4.1 Configuration (`main.py:10-14`)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `POP_SIZE` | 5 | Population size |
| `GENERATIONS` | 100 | Number of evolutionary generations |
| `EPOCHS_PER_GENOME` | 10 | Training epochs per fitness evaluation |
| `BATCH_SIZE` | 256 | Training batch size |

#### 4.2 Initialization (`main.py:16-40`)
| Step | Lines | Operation |
|------|-------|-----------|
| Device detection | 16-17 | `get_device()` from utils |
| Data loading | 20-21 | `get_data_loaders(batch_size=BATCH_SIZE)` |
| Population creation | 24-25 | `Population(size=POP_SIZE)` |
| Ray initialization | 28-31 | `ray.init()` with Windows fork workaround |
| Actor creation | 35-36 | Creates 5 `TrainActor` instances |

#### 4.3 Evolutionary Loop (`main.py:44-96`)

**Distributed Training (lines 47-65):**
| Step | Lines | Operation |
|------|-------|-----------|
| Task dispatch | 48-54 | Round-robin assignment to actors via `actor.train.remote()` |
| Result collection | 57-65 | `ray.get(future)` with exception handling for crashed actors |

**Fitness Calculation (lines 68-78):**
```
fitness = val_acc - 0.005 * log(param_count)
```
- Validation accuracy from training result (line 69)
- Parameter count penalty using logarithm (lines 72-74)
- Stored in `genome.fitness` at line 76

**History Tracking (lines 80-90):**
- Computes `best_fit` and `avg_fit` per generation
- Appends to `history` list
- Writes to `evolution_history.csv` at line 89

**Evolution (lines 92-94):**
- Calls `pop.evolve()` for all generations except the last

#### 4.4 Best Model Handling (`main.py:97-141`)
| Step | Lines | Operation |
|------|-------|-----------|
| Find best | 97 | `max(pop.individuals, key=lambda x: x.fitness)` |
| Decode model | 106-107 | `best_genome.decode()` |
| Save genome | 112-115 | Write genes to `best_genome.json` |
| Retrain | 120-127 | Train best genome for 50 epochs with `return_model=True` |
| Save weights | 129 | `torch.save(trained_model.state_dict(), 'best_model.pth')` |

---

### 5. Utility Functions (`utils.py:1-162`)

#### 5.1 Device Detection (`utils.py:13-59`)
The `get_device()` function implements priority-based device selection:

| Priority | Lines | Device | Detection Method |
|----------|-------|--------|------------------|
| 1 | 18-37 | CUDA | `torch.cuda.is_available()` + smoke test |
| 2 | 39-42 | MPS (Apple) | `torch.backends.mps.is_available()` |
| 3 | 44-57 | DirectML | `import torch_directml` |
| 4 | 55 | CPU | Fallback |

**CUDA Smoke Test (lines 22-24):** Creates tensor, moves to GPU, performs multiplication, moves back to CPU. Exits with code 1 on failure (line 37).

#### 5.2 Checkpointing (`utils.py:65-116`)

**`save_checkpoint()` (lines 65-76):**
- Creates checkpoint directory if needed
- Saves state dict to `checkpoint.pth`
- If `is_best=True`, copies to `model_best.pth`

**`load_checkpoint()` (lines 78-116):**
- Loads checkpoint from path
- Looks for `state_dict` or `model_state_dict` keys
- Attempts strict loading, falls back to non-strict on failure (lines 100-104)
- Optionally restores optimizer and scheduler states

#### 5.3 Visualization (`utils.py:122-161`)
The `plot_training_metrics()` function:
- Creates 1 or 2 subplot figure based on whether metrics are provided
- Plots train/val loss on first subplot
- Plots optional metric (e.g., accuracy) on second subplot
- Saves as PNG with 300 DPI to specified directory

---

### 6. Dashboard (`dashboard.py:1-61`)

Streamlit-based real-time monitoring application.

#### 6.1 Layout (`dashboard.py:11-22`)
- Two-column layout: 2:1 ratio
- Left column: Fitness evolution line chart
- Right column: Current status metrics + best architecture JSON

#### 6.2 Data Loading Functions
| Function | Lines | Purpose |
|----------|-------|---------|
| `load_data()` | 24-27 | Reads `evolution_history.csv` if exists |
| `load_best_genome()` | 29-33 | Reads `best_genome_golden.json` if exists |

#### 6.3 Auto-Refresh Loop (`dashboard.py:36-60`)
- Runs infinite `while True` loop
- Updates chart with `best_fitness` and `avg_fitness` columns (line 42)
- Displays best fitness metric (line 49)
- Shows best genome JSON (lines 52-54)
- Refreshes every 2 seconds (line 60)

---

## Data Flow

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                         main.py                             │
                                    └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌───────────────────────┐       ┌─────────────────────────────────────────────────────────────┐
│   Population.py       │◀──────│  pop = Population(size=POP_SIZE)                            │
│   Creates N genomes   │       │  Initializes N random Genome objects                        │
└───────────────────────┘       └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │  FOR each generation (0 to GENERATIONS-1):                  │
                                    └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌───────────────────────┐       ┌─────────────────────────────────────────────────────────────┐
│   TrainActor (Ray)    │◀──────│  actor.train.remote(genome, epochs=EPOCHS_PER_GENOME)       │
│   Parallel training   │       │  Distributed across 5 actors with 0.2 GPU each              │
└───────────────────────┘       └─────────────────────────────────────────────────────────────┘
         │                                                     │
         ▼                                                     │
┌───────────────────────┐                                      │
│   genome.decode()     │                                      │
│   genes → nn.Module   │                                      │
└───────────────────────┘                                      │
         │                                                     │
         ▼                                                     │
┌───────────────────────┐                                      │
│   train_model()       │                                      │
│   Adam optimizer      │                                      │
│   CrossEntropyLoss    │                                      │
│   Returns (acc, size) │                                      │
└───────────────────────┘                                      │
                                                               ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │  fitness = val_acc - 0.005 * log(param_count)               │
                                    └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌───────────────────────┐       ┌─────────────────────────────────────────────────────────────┐
│   evolution_history   │◀──────│  pd.DataFrame(history).to_csv('evolution_history.csv')      │
│   .csv                │       │  Logged after each generation                                │
└───────────────────────┘       └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌───────────────────────┐       ┌─────────────────────────────────────────────────────────────┐
│   Population.evolve() │◀──────│  pop.evolve() - Selection, Crossover, Mutation              │
│   Creates next gen    │       │  Elitism: Top 2 carried over                                 │
└───────────────────────┘       └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │  END FOR                                                     │
                                    └─────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌───────────────────────┐       ┌─────────────────────────────────────────────────────────────┐
│   best_genome.json    │◀──────│  Save best genome genes                                      │
│   best_model.pth      │◀──────│  Retrain best for 50 epochs, save weights                    │
└───────────────────────┘       └─────────────────────────────────────────────────────────────┘
```

---

## Key Patterns

### Genetic Algorithm Pattern
- **Encoding**: `Genome` class represents solutions as gene dictionaries (`genome.py:7-259`)
- **Decoding**: `decode()` method phenotype expression (`genome.py:66-173`)
- **Selection**: Tournament selection with `k=3` (`population.py:46-51`)
- **Crossover**: Variable split crossover (`population.py:53-90`)
- **Mutation**: Multiple mutation operators (`genome.py:175-252`)
- **Elitism**: Top 2 individuals preserved (`population.py:31`)

### Actor Model (Ray)
- **Distributed Computation**: `TrainActor` remote class (`trainer.py:126-144`)
- **Resource Management**: Fractional GPU allocation `num_gpus=0.2` (`trainer.py:126`)
- **Fault Tolerance**: `max_restarts=-1` for automatic recovery (`trainer.py:126`)
- **Round-Robin Scheduling**: `actors[i % num_actors]` (`main.py:49`)

### Dummy Forward Pass Pattern
- **Purpose**: Dynamically compute flattened feature size
- **Location**: `genome.py:140-148`
- **Mechanism**: Creates zero tensor, passes through feature extractor, measures output size

---

## Configuration

### Hyperparameters (`main.py:10-14`)
| Parameter | Value | Location |
|-----------|-------|----------|
| Population Size | 5 | `main.py:11` |
| Generations | 100 | `main.py:12` |
| Epochs per Genome | 10 | `main.py:13` |
| Batch Size | 256 | `main.py:14` |

### Evolutionary Parameters
| Parameter | Value | Location |
|-----------|-------|----------|
| Elitism Count | 2 | `population.py:17` (default) |
| Mutation Rate | 0.1 | `population.py:17` (default) |
| Tournament Size | 3 | `population.py:46` (default) |

### Architecture Search Space (`genome.py:38-55`)
| Layer Type | Options |
|------------|---------|
| Conv2d channels | `[16, 32, 64, 128]` |
| Conv2d kernel | `[3, 5]` |
| Dropout probability | `[0.1, 0.3, 0.5]` |
| Learning rate | `[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]` |
| Conv blocks | `1-3` |

### Data Normalization (`trainer.py:17, 22`)
| Component | Mean | Std |
|-----------|------|-----|
| Red | 0.4914 | 0.2023 |
| Green | 0.4822 | 0.1994 |
| Blue | 0.4465 | 0.2010 |

---

## Error Handling

| Location | Error Type | Handling |
|----------|------------|----------|
| `genome.py:145-148` | Invalid architecture | Raises `ValueError`, caught in trainer |
| `trainer.py:50-58` | Genome decode failure | Returns `(0.0, 0)` for fitness |
| `main.py:59-65` | Actor crash | `ray.exceptions.ActorUnavailableError` → fitness 0 |
| `main.py:63-65` | Generic training error | Exception caught → fitness 0 |
| `main.py:137-139` | Model save failure | Exception caught and printed |
| `utils.py:31-37` | CUDA smoke test failure | `sys.exit(1)` |
| `utils.py:102-104` | Checkpoint load mismatch | Falls back to `strict=False` |

---

## Artifacts

| File | Content | Created By |
|------|---------|------------|
| `evolution_history.csv` | Per-generation fitness statistics | `main.py:89` |
| `best_genome.json` | Best genome's gene list | `main.py:113-115` |
| `best_model.pth` | Best model's state dict | `main.py:129` |
| `best_genome_golden.json` | Pre-saved best genome | Manual/previous run |
| `best_model_golden.pth` | Pre-saved best model weights | Manual/previous run |
| `data/` | CIFAR-10 dataset | `trainer.py:26,29` |

---

## Module Dependencies

```
main.py
├── torch
├── math
├── json
├── pandas
├── ray
└── neuroevolve/
    ├── population.py
    │   └── genome.py (Genome class)
    ├── utils.py (get_device)
    └── trainer.py
        ├── get_data_loaders
        ├── train_model
        └── TrainActor (Ray actor)
            └── genome.py (Genome class)

dashboard.py
├── streamlit
├── pandas
├── json
├── time
└── os
```
