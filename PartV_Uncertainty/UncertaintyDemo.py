# uncertainty_viz.py
import torch
import matplotlib.pyplot as plt


class TokenUncertaintyVisualizer:
    def __init__(self, model, tokenizer, prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt

        self.generated_tokens = None
        self.generated_text = None
        self.logits = None  # [num_generated_tokens, vocab_size]

    def token_entropy(self, logits):
      probs = torch.softmax(self.logits, dim=-1)
      entropy = -torch.sum(probs * torch.log(probs + 1e-12))
      return entropy

    def generate_and_store_output(
        self,
        max_new_tokens=80,
        do_sample=True,
        temperature=1.0
    ):
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                output_scores=True,
                return_dict_in_generate=True
            )

        self.generated_tokens = outputs.sequences[0]
        self.logits = torch.stack(outputs.scores, dim=0)

        self.generated_text = self.tokenizer.decode(
            self.generated_tokens,
            skip_special_tokens=True
        )

        print("\n--- Generated Output ---")
        print(self.generated_text)

        print("\n--- Generated Tokens (index : token) ---")
        gen_token_ids = self.generated_tokens[-self.logits.size(0):]
        tokens = self.tokenizer.convert_ids_to_tokens(gen_token_ids.tolist())
        
        for idx, tok in enumerate(tokens):
            entropy = self.token_entropy(self.logits[idx]) if idx < len(self.logits) else 0.0
            print(f"{idx:2d} : {repr(tok)} : entropy = {entropy}")

    def plot_token_distribution_at_position(self, token_index, top_k=100):
        if self.logits is None:
            raise ValueError("Call generate_and_store_output() first.")

        if not (0 <= token_index < self.logits.size(0)):
            raise IndexError("token_index out of range.")

        token_logits = self.logits[token_index]
        probs = torch.softmax(token_logits, dim=-1)

        top_probs, top_ids = torch.topk(probs, k=top_k)

        # ✅ CRITICAL FIX: flatten explicitly
        top_probs = top_probs.detach().cpu().view(-1).numpy()
        top_ids = top_ids.detach().cpu().view(-1).tolist()

        top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids)

        # Actual generated token
        actual_token_id = self.generated_tokens[
            -self.logits.size(0) + token_index
        ].item()
        actual_token = self.tokenizer.convert_ids_to_tokens(actual_token_id)

        plt.figure(figsize=(10, 4))
        plt.bar(range(top_k), top_probs)

        # Highlight actual sampled token if present
        if actual_token_id in top_ids:
            idx = top_ids.index(actual_token_id)
            plt.bar(idx, top_probs[idx], edgecolor="red", linewidth=2)

        plt.title(
            f"Token probability distribution at position {token_index}\n"
            f"Sampled token: {repr(actual_token)}"
        )
        plt.ylabel("Probability")
        plt.ylim(0, 1.0)
        plt.xticks([])
        plt.tight_layout()
        plt.show()
