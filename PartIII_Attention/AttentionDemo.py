# attention_demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import ipywidgets as widgets
from IPython.display import display

# This code was generated with the help of an AI language model.

class AttentionDemo:
    """
    Interactive attention playground using the first encoder layer of a pretrained BERT.
    Usage:
        from attention_demo import AttentionDemo
        AttentionDemo()    # runs in-place, prompts for input, then shows widgets
    """
    def __init__(self, model_name="bert-base-uncased", device=None, tokenizer=None, model=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer and model ({model_name}) on {self.device} — this may take a moment...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(self.device) if model is None else model
        self.model.eval()

        cfg = self.model.config
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads

        # Run interactive flow
        self.run_demo()

    def run_demo(self):
        text = input("Enter a short sentence to inspect attention (keep it short):\n")
        if not text.strip():
            print("Empty input — aborting.")
            return

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        self.input_ids = inputs["input_ids"].to(self.device)   # shape (1, seq_len)
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[0].cpu().tolist())

        # Compute embeddings and first-layer attention Q/K/V without running the full forward pass
        with torch.no_grad():
            # embeddings: (1, seq_len, hidden_size)
            embeddings = self.model.embeddings(self.input_ids)

            # First encoder layer (BERT structure: model.encoder.layer[0])
            layer0 = self.model.encoder.layer[0]
            # Linear projections to get Q, K, V (shapes: (1, seq_len, hidden_size))
            mixed_query = layer0.attention.self.query(embeddings)
            mixed_key   = layer0.attention.self.key(embeddings)
            mixed_value = layer0.attention.self.value(embeddings)

            # Reshape to (batch, num_heads, seq_len, head_dim)
            def transpose_for_scores(x):
                b, seq_len, hidden = x.size()
                x = x.view(b, seq_len, self.num_heads, self.head_dim)
                return x.permute(0, 2, 1, 3)

            q = transpose_for_scores(mixed_query)   # (1, heads, seq_len, head_dim)
            k = transpose_for_scores(mixed_key)
            v = transpose_for_scores(mixed_value)

            # Attention scores: (1, heads, seq_len, seq_len)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            probs = torch.softmax(scores, dim=-1)

            # Move to CPU numpy for visualization
            self.attention_probs = probs.squeeze(0).cpu().numpy()   # (heads, seq_len, seq_len)
            self.v_per_head = v.squeeze(0).cpu().numpy()           # (heads, seq_len, head_dim)

        # Launch widgets/plots
        self._build_ui()

    def _build_ui(self):
        seq_len = len(self.tokens)
        print("\nTokens (idx : token):")
        for i, t in enumerate(self.tokens):
            print(f"{i:2d}: {t}")

        head_slider = widgets.IntSlider(value=0, min=0, max=self.num_heads-1, step=1, description="Head")
        token_slider = widgets.IntSlider(value=0, min=0, max=seq_len-1, step=1, description="Query idx")
        topk_slider = widgets.IntSlider(value=5, min=1, max=min(20, seq_len), step=1, description="Top-k")

        controls = widgets.HBox([head_slider, token_slider, topk_slider])
        out = widgets.Output()
        display(controls, out)

        def render(change=None):
            head = head_slider.value
            qidx = token_slider.value
            topk = topk_slider.value

            with out:
                out.clear_output(wait=True)

                weights = self.attention_probs[head, qidx]  # (seq_len,)
                # Top-k keys (indices) by attention weight
                topk_idx = np.argsort(weights)[-topk:][::-1]

                # Bar plot of attention weights (query -> all keys)
                fig, axes = plt.subplots(1, 2, figsize=(12, 3))
                ax_bar, ax_mat = axes

                ax_bar.bar(range(seq_len), weights)
                ax_bar.set_xticks(range(seq_len))
                ax_bar.set_xticklabels(self.tokens, rotation=45, ha="right", fontsize=9)
                ax_bar.set_title(f'Attention weights for query token idx={qidx} ("{self.tokens[qidx]}"), head={head}')
                ax_bar.set_ylim(0, weights.max()*1.05)

                # Highlight top-k bars
                for idx in topk_idx:
                    ax_bar.get_children()[idx].set_edgecolor("red")
                    ax_bar.get_children()[idx].set_linewidth(1.5)

                # Full attention matrix heatmap for this head
                im = ax_mat.imshow(self.attention_probs[head], aspect='auto', cmap='viridis')
                ax_mat.set_xticks(range(seq_len)); ax_mat.set_xticklabels(self.tokens, rotation=45, ha='right', fontsize=9)
                ax_mat.set_yticks(range(seq_len)); ax_mat.set_yticklabels(self.tokens, fontsize=9)
                ax_mat.set_title(f'Full attention matrix (head {head})')
                plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.show()

                # Compute aggregated value vector (weighted sum of V for this head, for the chosen query)
                v_head = self.v_per_head[head]   # (seq_len, head_dim)
                agg = (weights[:, None] * v_head).sum(axis=0)   # (head_dim,)

                print(f"\nTop-{topk} attended tokens (by weight):")
                for rank, idx in enumerate(topk_idx, start=1):
                    print(f" {rank}. idx={idx:2d} token={self.tokens[idx]!r} weight={weights[idx]:.4f}")

                # Print first few dimensions of aggregated vector (rounded)
                print("\nAggregated value vector for this query (first 12 dims):")
                print(np.round(agg[:12], 4))

        # Wire up interactions
        head_slider.observe(render, names='value')
        token_slider.observe(render, names='value')
        topk_slider.observe(render, names='value')

        # initial render
        render()
