from transformers import AutoTokenizer
import matplotlib.pyplot as plt


class TokenizationDemo:
    def __init__(self, tokenizer_name="gpt2"):
        """
        Initialize tokenizer once so the notebook stays clean.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT-2 has no pad token; not needed here but avoids warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.run_demo()

    def run_demo(self):
        """
        Prompt user, tokenize input, and visualize tokens.
        """
        text = input("Enter a string to tokenize:\n")

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        self.display_summary(text, tokens, token_ids)
        self.visualize_tokens(tokens)

    def display_summary(self, text, tokens, token_ids):
        """
        Print concise textual summary.
        """
        print("\n--- Input Summary ---")
        print(f"Raw text: {repr(text)}")
        print(f"Character count: {len(text)}")
        print(f"Token count: {len(tokens)}\n")

        print("Tokens and IDs:")
        for tok, tid in zip(tokens, token_ids):
            print(f"{tok:>12}  →  {tid}")

    def visualize_tokens(self, tokens):
        """
        Display tokens as colored boxes for visual intuition.
        """
        fig, ax = plt.subplots(figsize=(max(8, len(tokens)), 2))
        ax.axis("off")

        x = 0
        for tok in tokens:
            ax.text(
                x, 0.5, tok,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#dbeafe"),
                ha="left", va="center"
            )
            x += len(tok) + 1

        ax.set_title("Tokenized Representation", fontsize=14)
        plt.show()
