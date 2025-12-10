import torch
import torch.nn as nn
import random
import copy
from typing import List, Dict, Any, Optional

class Genome:
    def __init__(self, genes: Optional[List[Dict[str, Any]]] = None, learning_rate: float = None):
        """
        Initialize a Genome.
        
        Args:
            genes: A list of dictionaries, where each dictionary represents a layer configuration.
                   If None, initializes a random minimal architecture.
            learning_rate: The learning rate for the optimizer.
        """
        self.fitness: float = 0.0
        if genes is not None:
            self.genes = genes
        else:
            self.genes = self._create_random_genes()
            
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            # Expanded learning rate options for diversity
            self.learning_rate = random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

    def _create_random_genes(self) -> List[Dict[str, Any]]:
        """Creates a random initial architecture."""
        genes = []
        
        # Randomly decide depth (1 to 3 conv blocks)
        num_blocks = random.randint(1, 3)
        
        for _ in range(num_blocks):
            # Conv Layer
            out_channels = random.choice([16, 32, 64, 128])
            kernel_size = random.choice([3, 5])
            padding = kernel_size // 2
            
            genes.append({
                'type': 'Conv2d', 
                'out_channels': out_channels, 
                'kernel_size': kernel_size, 
                'stride': 1, 
                'padding': padding
            })
            
            # Activation
            genes.append({'type': 'ReLU'})
            
            # Optional Dropout
            if random.random() < 0.3:
                genes.append({'type': 'Dropout', 'p': random.choice([0.1, 0.3, 0.5])})
            
            # Pooling (always add to reduce dimensionality)
            genes.append({'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2})
            
        # Flatten and Head
        genes.append({'type': 'Flatten'})
        genes.append({'type': 'Linear', 'out_features': 10}) # CIFAR-10 has 10 classes
        
        return genes

    def decode(self, input_shape: tuple = (3, 32, 32)) -> nn.Module:
        """
        Converts the genome into a PyTorch model.
        
        Args:
            input_shape: The shape of the input tensor (C, H, W).
            
        Returns:
            A torch.nn.Module representing the architecture.
        """
        layers = []
        current_channels = input_shape[0]
        current_height = input_shape[1]
        current_width = input_shape[2]
        
        # We use a dummy forward pass approach for the Flatten->Linear transition 
        # because it's robust and easy to implement.
        
        # We will build the feature extractor first
        feature_extractor_layers = []
        classifier_layers = []
        
        is_flattened = False
        
        for gene in self.genes:
            layer_type = gene['type']
            
            if layer_type == 'Conv2d':
                out_channels = gene['out_channels']
                kernel_size = gene['kernel_size']
                stride = gene.get('stride', 1)
                padding = gene.get('padding', 1) # Default 'same' padding behavior for k=3, s=1
                
                layer = nn.Conv2d(current_channels, out_channels, kernel_size, stride, padding)
                feature_extractor_layers.append(layer)
                current_channels = out_channels
                # Update spatial dims (approximate if padding is 'same' logic, but let's trust the dummy pass)
                
            elif layer_type == 'BatchNorm2d':
                feature_extractor_layers.append(nn.BatchNorm2d(current_channels))
                
            elif layer_type == 'ReLU':
                feature_extractor_layers.append(nn.ReLU())
                
            elif layer_type == 'MaxPool2d':
                kernel_size = gene['kernel_size']
                stride = gene['stride']
                feature_extractor_layers.append(nn.MaxPool2d(kernel_size, stride))
                
            elif layer_type == 'Dropout':
                p = gene.get('p', 0.5)
                # Dropout can be in features or classifier
                if is_flattened:
                    classifier_layers.append(nn.Dropout(p))
                else:
                    feature_extractor_layers.append(nn.Dropout2d(p))
                    
            elif layer_type == 'Flatten':
                is_flattened = True
                
            elif layer_type == 'Linear':
                out_features = gene['out_features']
                # We need to know input features. 
                # We'll handle this connection when assembling.
                classifier_layers.append(nn.Linear(1, out_features)) # Placeholder in_features
                
        # Assemble
        model = nn.Sequential()
        
        # Add feature extractor
        features = nn.Sequential(*feature_extractor_layers)
        model.add_module('features', features)
        
        # Calculate flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            try:
                dummy_out = features(dummy_input)
                flatten_size = dummy_out.view(1, -1).size(1)
            except Exception as e:
                # If architecture is invalid (e.g. pools too much), return a dummy valid model or raise
                # For now, let's raise so we know to fix the mutation logic or catch it
                raise ValueError(f"Invalid architecture generated: {e}")

        model.add_module('flatten', nn.Flatten())
        
        # Add classifier
        # Fix the first Linear layer's input size
        if len(classifier_layers) > 0:
            first_linear = classifier_layers[0]
            if isinstance(first_linear, nn.Linear):
                # Replace with correct in_features
                classifier_layers[0] = nn.Linear(flatten_size, first_linear.out_features)
            
            # Fix subsequent linear layers if any (not implemented in random init but good for future)
            current_features = classifier_layers[0].out_features
            for i in range(1, len(classifier_layers)):
                if isinstance(classifier_layers[i], nn.Linear):
                    classifier_layers[i] = nn.Linear(current_features, classifier_layers[i].out_features)
                    current_features = classifier_layers[i].out_features

            classifier = nn.Sequential(*classifier_layers)
            model.add_module('classifier', classifier)
        else:
            # Fallback if no linear layer (shouldn't happen with valid genes)
            model.add_module('classifier', nn.Linear(flatten_size, 10))

        return model

    def mutate(self, mutation_rate: float = 0.1):
        """
        Applies random mutations to the genome.
        """
        if random.random() < mutation_rate:
            mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_param', 'modify_lr'])
            
            if mutation_type == 'add_layer':
                self._add_random_layer()
            elif mutation_type == 'remove_layer':
                self._remove_random_layer()
            elif mutation_type == 'modify_param':
                self._modify_random_param()
            elif mutation_type == 'modify_lr':
                self.learning_rate = random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

    def _add_random_layer(self):
        # Insert a layer at a random position before Flatten
        # Find index of Flatten
        flatten_idx = -1
        for i, gene in enumerate(self.genes):
            if gene['type'] == 'Flatten':
                flatten_idx = i
                break
        
        if flatten_idx == -1: return # Should not happen
        
        insert_idx = random.randint(0, flatten_idx)
        
        layer_type = random.choice(['Conv2d', 'BatchNorm2d', 'ReLU', 'Dropout'])
        
        new_gene = {}
        if layer_type == 'Conv2d':
            new_gene = {
                'type': 'Conv2d',
                'out_channels': random.choice([16, 32, 64, 128]),
                'kernel_size': random.choice([3, 5]),
                'stride': 1,
                'padding': 1 # Simplified
            }
        elif layer_type == 'BatchNorm2d':
            new_gene = {'type': 'BatchNorm2d'}
        elif layer_type == 'ReLU':
            new_gene = {'type': 'ReLU'}
        elif layer_type == 'Dropout':
            new_gene = {'type': 'Dropout', 'p': random.choice([0.1, 0.3, 0.5])}
            
        self.genes.insert(insert_idx, new_gene)

    def _remove_random_layer(self):
        # Remove a random layer before Flatten, ensuring we don't remove everything
        flatten_idx = -1
        for i, gene in enumerate(self.genes):
            if gene['type'] == 'Flatten':
                flatten_idx = i
                break
                
        if flatten_idx <= 1: return # Don't remove if too few layers
        
        remove_idx = random.randint(0, flatten_idx - 1)
        self.genes.pop(remove_idx)

    def _modify_random_param(self):
        # Pick a random gene and modify a parameter
        if not self.genes: return
        
        idx = random.randint(0, len(self.genes) - 1)
        gene = self.genes[idx]
        
        if gene['type'] == 'Conv2d':
            param = random.choice(['out_channels', 'kernel_size'])
            if param == 'out_channels':
                gene['out_channels'] = random.choice([16, 32, 64, 128])
            elif param == 'kernel_size':
                gene['kernel_size'] = random.choice([3, 5])
                gene['padding'] = gene['kernel_size'] // 2 # Keep padding consistent
        elif gene['type'] == 'Dropout':
            gene['p'] = random.choice([0.1, 0.3, 0.5])

    def copy(self):
        """Returns a deep copy of the genome."""
        new_genome = Genome(genes=copy.deepcopy(self.genes), learning_rate=self.learning_rate)
        new_genome.fitness = self.fitness
        return new_genome
