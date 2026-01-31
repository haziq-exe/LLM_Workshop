import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA


class EmbeddingDemo:
    def __init__(self, model_name="gpt2", tokenizer=None):
        """
        Loads tokenizer and embedding table once.
        """
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        model = AutoModel.from_pretrained(model_name)

        # Input embedding matrix: [vocab_size, hidden_dim]
        self.embedding_table = model.get_input_embeddings().weight.detach()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.run_demo()

    def run_demo(self):
        """
        Main interactive flow.
        """
        word = input("Enter a word to inspect its embedding:\n")

        token_id = self.tokenizer.encode(word, add_special_tokens=False)

        if len(token_id) == 0:
            print("No tokens produced.")
            return

        token_id = token_id[0]
        token = self.tokenizer.decode([token_id])
        embedding = self.embedding_table[token_id]

        self.display_embedding_info(token, token_id, embedding)
        self.run_geometry_demo()

    def display_embedding_info(self, token, token_id, embedding):
        """
        Show that token IDs are just lookup indices.
        """
        print("\n--- Token → Embedding Lookup ---")
        print(f"Token: {repr(token)}")
        print(f"Token ID: {token_id}")
        print(f"Embedding dimension: {embedding.shape[0]}")
        print(f"Embedding norm: {embedding.norm().item():.4f}")
        print("\nFirst 10 embedding values:")
        print(embedding[:10].numpy())

    def run_geometry_demo(self):
        """
        Show meaning lives in geometry, not IDs.
        """
        words = ["man", "woman", "king", "queen", "apple"]
        token_ids = [
            self.tokenizer.encode(w, add_special_tokens=False)[0]
            for w in words
        ]

        embeddings = self.embedding_table[token_ids].numpy()

        self.plot_embedding_space(words, embeddings)
        self.vector_arithmetic_demo(words, embeddings)

    def plot_embedding_space(self, words, embeddings):
        """
        2D PCA projection of embeddings.
        """
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        plt.figure(figsize=(5, 5))
        for word, (x, y) in zip(words, reduced):
            plt.scatter(x, y)
            plt.text(x + 0.02, y + 0.02, word, fontsize=10)

        plt.title("Embedding Space (PCA Projection)")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        plt.show()

    def vector_arithmetic_demo(self, words, embeddings):
        """
        king - man + woman ≈ ?
        """
        word_to_vec = dict(zip(words, embeddings))

        target_vec = (
            word_to_vec["king"]
            - word_to_vec["man"]
            + word_to_vec["woman"]
        )

        # Cosine similarity against full vocab
        norms = self.embedding_table.norm(dim=1)
        sims = torch.matmul(
            self.embedding_table, torch.tensor(target_vec)
        ) / (norms * torch.norm(torch.tensor(target_vec)))

        top_ids = torch.topk(sims, k=5).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]) for i in top_ids]

        print("\n--- Vector Arithmetic ---")
        print("king - man + woman ≈")
        for tok in top_tokens:
            print(f"  {repr(tok)}")
