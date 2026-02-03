from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

class SuperWeightDemo:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", 
                 phi_super_coords = [
    (2, 525, 808),
    (2, 1693, 808),
    (2, 1113, 808),
    (4, 525, 2723),
    (4, 1113, 2723),
    (4, 1693, 2723),]):
        self.model_name = model_name
        self.coords = phi_super_coords

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def prune_phi3_super_weights(self):
        """
        Zero out target weights in Phi‑3 MLP down_proj layers.
        coords: list of (layer_idx, row, col)
        """
        with torch.no_grad():
            for layer_idx, row, col in self.coords:
                # Access the Phi3MLP inside each decoder layer
                mlp = self.model.model.layers[layer_idx].mlp
                # down_proj weight is (intermediate_size, hidden_size)
                W = mlp.down_proj.weight
                print(f"Before: {W[row, col].item():.6f}")
                W[row, col] = 0.0
                print(f"After : {W[row, col].item():.6f}")

    def test_model_output(self, prompt="Once upon a time, "):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        print(self.tokenizer.decode(self.model.generate(**inputs, max_new_tokens=20)[0]))


    def restore_super_and_prune_random(self, fraction=0.001, seed=42):
        """
        Restore super weights and prune a small fraction of other weights randomly.
        
        model: Phi‑3‑mini‑4k‑instruct model
        super_coords: list of (layer_idx, row, col) tuples
        fraction: fraction of non-super weights to prune
        seed: random seed for reproducibility
        """
        random.seed(seed)
        torch.manual_seed(seed)

        # Backup super weights
        super_values = {}
        with torch.no_grad():
            for layer_idx, row, col in self.coords:
                W = self.model.model.layers[layer_idx].mlp.down_proj.weight
                super_values[(layer_idx, row, col)] = W[row, col].item()
                # Restore the original super weight
                W[row, col] = super_values[(layer_idx, row, col)]

        # Convert super_coords to a set for fast lookup
        super_set = set(self.coords)

        # Prune a fraction of other weights
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.model.layers):
                W = layer.mlp.down_proj.weight
                n_rows, n_cols = W.shape
                total_weights = n_rows * n_cols
                n_prune = int(total_weights * fraction)

                pruned = 0
                while pruned < n_prune:
                    row = random.randint(0, n_rows - 1)
                    col = random.randint(0, n_cols - 1)
                    if (layer_idx, row, col) not in super_set and W[row, col].item() != 0.0:
                        W[row, col] = 0.0
                        pruned += 1
                print(f"Layer {layer_idx}: Restored super weights and pruned {pruned} random weights")


    def prune_random_except_super(self, fraction=0.01, seed=42):
        """
        Prune a small fraction of weights randomly, but leave super weights intact.
        
        model: the Phi‑3‑mini‑4k‑instruct model
        super_coords: list of (layer_idx, row, col) tuples
        fraction: fraction of weights to zero randomly
        seed: random seed for reproducibility
        """
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Convert super_coords to a set for fast lookup
        super_set = set(self.coords)
        
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.model.layers):
                W = layer.mlp.down_proj.weight
                n_rows, n_cols = W.shape
                total_weights = n_rows * n_cols
                n_prune = int(total_weights * fraction)
                
                pruned = 0
                while pruned < n_prune:
                    row = random.randint(0, n_rows - 1)
                    col = random.randint(0, n_cols - 1)
                    if (layer_idx, row, col) not in super_set and W[row, col].item() != 0.0:
                        W[row, col] = 0.0
                        pruned += 1
                print(f"Layer {layer_idx}: Pruned {pruned} random weights, super weights preserved")
