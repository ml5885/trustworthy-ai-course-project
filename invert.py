import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 32
BATCH_SIZE = 16
NUM_STEPS = 5000

BLOCK_LAYER = 4                # transformer block index
LAYER_IDX = BLOCK_LAYER + 1    # index into hidden_states (0 = embeddings)
STEERING_SCALE = 9.0

TASK = "sequence"      # "token" or "sequence"


STEERING_SENTENCES = [
    "The old clock on the wall ticked loudly.",
    "She collected seashells every summer at the beach.",
    "The cat jumped onto the windowsill to watch birds.",
    "His favorite ice cream flavor was mint chocolate chip.",
    "The book fell open to page 217.",
    "Lightning flashed across the night sky.",
    "They planted tulip bulbs in the garden last fall.",
    "The coffee shop was bustling with morning customers.",
    "She tied her hiking boots with double knots.",
    "The museum exhibit featured ancient Egyptian artifacts.",
    "Children laughed as they ran through the sprinkler.",
    "The train arrived precisely on schedule.",
    "He couldn't remember where he had parked his car.",
    "Autumn leaves crunched beneath their feet.",
    "The recipe called for two teaspoons of vanilla extract.",
    "The dog wagged its tail excitedly at the park.",
    "Mountains loomed in the distance, covered with snow.",
    "She practiced piano for three hours every day.",
    "The telescope revealed stunning details of Saturn's rings.",
    "Fresh bread was baking in the oven.",
    "They watched the sunset from the rooftop.",
    "The professor explained the theory with great enthusiasm.",
    "Waves crashed against the rocky shoreline.",
    "He assembled the furniture without reading the instructions.",
    "Stars twinkled brightly in the clear night sky.",
    "The old photograph brought back forgotten memories.",
    "Bees buzzed around the flowering cherry tree.",
    "She solved the crossword puzzle in record time.",
    "The air conditioner hummed quietly in the background.",
    "Rain pattered softly against the windowpane.",
    "The movie theater was packed for the premiere.",
    "He sketched the landscape with charcoal pencils.",
    "Children built sandcastles at the water's edge.",
    "The orchestra tuned their instruments before the concert.",
    "Fragrant lilacs bloomed along the garden fence.",
    "The basketball bounced off the rim.",
    "She wrapped the birthday present with blue ribbon.",
    "The hiker followed the trail markers through the forest.",
    "Their canoe glided silently across the still lake.",
    "The antique vase was carefully wrapped in bubble wrap.",
    "Fireflies flickered in the summer twilight.",
    "The chef garnished the plate with fresh herbs.",
    "Wind chimes tinkled melodically on the porch.",
    "The flight attendant demonstrated safety procedures.",
    "He repaired the leaky faucet with a new washer.",
    "Fog shrouded the valley below the mountain.",
    "The comedian's joke made everyone laugh.",
    "She planted herbs in pots on the kitchen windowsill.",
    "The painting hung crookedly on the wall.",
    "Snowflakes drifted lazily from the gray sky."
]


def sample_batch(batch_size, seq_len, vocab_size, device):
    # This is the only place you need to touch to swap in a different training distribution.
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


class TokenProbe(nn.Module):
    """
    Hidden state -> token embedding -> logits via frozen LM embedding matrix.

    This makes the probe live in the same geometry as the base LM, so
    tokens with similar embeddings (e.g. different BPE chunkings of the
    same word) are naturally close in logit space.
    """
    def __init__(self, d_model, lm):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_model)

        emb = lm.get_input_embeddings().weight.detach()  # (V, d)
        self.register_buffer("token_emb", emb)           # not trainable

    def forward(self, h):
        # h: (N, d)
        v = self.fc2(self.act(self.fc1(h)))              # (N, d)
        logits = v @ self.token_emb.t()                  # (N, V)
        return logits


class SequenceInverter(nn.Module):
    def __init__(self, d_model, vocab_size, bos_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)
        self.bos_id = bos_id

    def forward(self, input_ids, h_cond):
        # input_ids: (B, T), h_cond: (B, d)
        B, T = input_ids.shape
        bos = torch.full((B, 1), self.bos_id,
                         device=input_ids.device, dtype=torch.long)
        prev_tokens = torch.cat([bos, input_ids[:, :-1]], dim=1)
        emb = self.embed(prev_tokens)                    # (B, T, d)
        h0 = h_cond.unsqueeze(0).contiguous()            # (1, B, d)  <-- fix hx not contiguous
        out, _ = self.gru(emb, h0)
        logits = self.out(out)                           # (B, T, V)
        return logits

    def generate(self, h_cond, max_len):
        B = h_cond.size(0)
        hidden = h_cond.unsqueeze(0).contiguous()        # (1, B, d)
        prev = torch.full((B, 1), self.bos_id,
                          device=h_cond.device, dtype=torch.long)
        generated = []
        for _ in range(max_len):
            emb = self.embed(prev)                       # (B, 1, d)
            out, hidden = self.gru(emb, hidden)
            logits = self.out(out[:, -1, :])             # (B, V)
            next_tok = logits.argmax(dim=-1)             # (B,)
            generated.append(next_tok)
            prev = next_tok.unsqueeze(1)
        return torch.stack(generated, dim=1)             # (B, max_len)


def compute_steering_vector(model, tok, layer_idx, device, sentences=STEERING_SENTENCES):
    print(f"Computing steering vector on hidden_states[{layer_idx}] from {len(sentences)} pairs...")
    diffs = []
    with torch.no_grad():
        for normal in sentences:
            caps = normal.upper()

            ids_norm = tok.encode(normal, return_tensors="pt").to(device)
            ids_caps = tok.encode(caps, return_tensors="pt").to(device)

            out_norm = model(input_ids=ids_norm, output_hidden_states=True)
            out_caps = model(input_ids=ids_caps, output_hidden_states=True)

            h_norm = out_norm.hidden_states[layer_idx][0]
            h_caps = out_caps.hidden_states[layer_idx][0]

            v_norm = h_norm[-1]
            v_caps = h_caps[-1]

            diff = (v_caps - v_norm).cpu()
            diffs.append(diff)

    diffs_tensor = torch.stack(diffs, dim=0)
    mean_diff = diffs_tensor.mean(dim=0)
    steering_vector = F.normalize(mean_diff, dim=0)
    pair_norms = diffs_tensor.norm(dim=1)
    avg_pair_norm = float(pair_norms.mean().item())

    print(
        f"Steering vector shape: {steering_vector.shape}, "
        f"avg_pair_norm: {avg_pair_norm:.4f}, "
        f"normalized_norm: {steering_vector.norm().item():.4f}"
    )

    return steering_vector.to(device), avg_pair_norm


def plot_curves(losses, grad_norms, prefix):
    os.makedirs("results", exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    royal_purple = "#7851A9"
    steps = range(len(losses))

    plt.figure()
    plt.plot(steps, losses, color=royal_purple)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"results/{prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(steps, grad_norms, color=royal_purple)
    plt.xlabel("Training step")
    plt.ylabel("Gradient norm")
    plt.tight_layout()
    plt.savefig(f"results/{prefix}_grad_norm.png")
    plt.close()


def train_token_probe(lm, tokenizer, device):
    d_model = lm.config.hidden_size
    vocab_size = lm.config.vocab_size

    probe = TokenProbe(d_model, lm).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    grad_norms = []

    for step in range(NUM_STEPS):
        input_ids = sample_batch(BATCH_SIZE, SEQ_LEN, vocab_size, device)

        with torch.no_grad():
            out = lm(input_ids=input_ids, output_hidden_states=True)
            h = out.hidden_states[LAYER_IDX]

        B, T, D = h.shape
        h_flat = h.view(B * T, D)
        targets = input_ids.view(B * T)

        logits = probe(h_flat)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()

        grad_norm_sq = 0.0
        for p in probe.parameters():
            grad_norm_sq += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm_sq ** 0.5

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)

        if step % 10 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            print(
                f"[token] step {step:04d}  loss {loss.item():.4f}  "
                f"acc {acc:.4f}  grad_norm {grad_norm:.4f}"
            )

    plot_curves(losses, grad_norms, "token")

    test_text = "Transformers are almost surely not injective."
    with torch.no_grad():
        enc = tokenizer(test_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        out = lm(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[LAYER_IDX][0]
        logits = probe(h)
        preds = logits.argmax(dim=-1)

    input_ids_list = input_ids[0].tolist()
    pred_ids_list = preds.tolist()
    input_tokens = [tokenizer.decode([tid]) for tid in input_ids_list]
    pred_tokens = [tokenizer.decode([pid]) for pid in pred_ids_list]

    final_input_id = input_ids_list[-1]
    final_input_token = tokenizer.decode([final_input_id])
    final_pred_id = pred_ids_list[-1]
    final_pred_token = tokenizer.decode([final_pred_id])

    print("=== token-level test (no steering) ===")
    print("text:", test_text)
    print("input_ids:", input_ids_list)
    print("predicted_ids:", pred_ids_list)
    print("input_tokens:", input_tokens)
    print("predicted_tokens:", pred_tokens)
    print("final_hidden_state_target_id:", final_input_id)
    print("final_hidden_state_pred_id:", final_pred_id)
    print("final_hidden_state_target_token:", final_input_token)
    print("final_hidden_state_pred_token:", final_pred_token)

    # steering-vector test
    print("\n=== steering-vector test (layer 4, scale 9) ===")
    steering_vector, avg_pair_norm = compute_steering_vector(
        lm, tokenizer, LAYER_IDX, device
    )

    with torch.no_grad():
        enc = tokenizer(test_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        out = lm(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[LAYER_IDX][0].clone()

        # add steering only to the final token's hidden state
        h[-1] = h[-1] + steering_vector * STEERING_SCALE

        logits_steered = probe(h)
        preds_steered = logits_steered.argmax(dim=-1)

    steered_ids_list = preds_steered.tolist()
    steered_tokens = [tokenizer.decode([tid]) for tid in steered_ids_list]

    print("steering_scale:", STEERING_SCALE)
    print("steered_predicted_ids:", steered_ids_list)
    print("steered_predicted_tokens:", steered_tokens)


def train_sequence_inverter(lm, tokenizer, device):
    d_model = lm.config.hidden_size
    vocab_size = lm.config.vocab_size
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0

    inverter = SequenceInverter(d_model, vocab_size, bos_id).to(device)
    optimizer = torch.optim.Adam(inverter.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    grad_norms = []

    for step in range(NUM_STEPS):
        input_ids = sample_batch(BATCH_SIZE, SEQ_LEN, vocab_size, device)

        with torch.no_grad():
            out = lm(input_ids=input_ids, output_hidden_states=True)
            h_cond = out.hidden_states[LAYER_IDX][:, -1, :]  # (B, d)

        logits = inverter(input_ids, h_cond)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), input_ids.view(B * T))

        optimizer.zero_grad()
        loss.backward()

        grad_norm_sq = 0.0
        for p in inverter.parameters():
            grad_norm_sq += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm_sq ** 0.5

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)

        if step % 10 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == input_ids).float().mean().item()
            print(
                f"[sequence] step {step:04d}  loss {loss.item():.4f}  "
                f"acc {acc:.4f}  grad_norm {grad_norm:.4f}"
            )

    plot_curves(losses, grad_norms, "sequence")

    test_text = "Transformers are almost surely not injective."
    with torch.no_grad():
        enc = tokenizer(test_text, return_tensors="pt")
        target_ids = enc["input_ids"].to(device)
        out = lm(input_ids=target_ids, output_hidden_states=True)
        h_target = out.hidden_states[LAYER_IDX][:, -1, :]

        max_len = target_ids.size(1)
        gen_ids = inverter.generate(h_target, max_len)

        out_gen = lm(input_ids=gen_ids, output_hidden_states=True)
        h_gen = out_gen.hidden_states[LAYER_IDX][:, -1, :]
        hidden_l2 = torch.norm(h_gen - h_target).item()

    target_ids_list = target_ids[0].tolist()
    target_tokens = [tokenizer.decode([tid]) for tid in target_ids_list]
    gen_ids_list = gen_ids[0].tolist()
    gen_tokens = [tokenizer.decode([gid]) for gid in gen_ids_list]

    print("=== sequence-level test ===")
    print("text:", test_text)
    print("target_ids:", target_ids_list)
    print("reconstructed_ids:", gen_ids_list)
    print("target_tokens:", target_tokens)
    print("reconstructed_tokens:", gen_tokens)
    print("final_hidden_state_L2_distance:", hidden_l2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if TASK == "token":
        train_token_probe(lm, tokenizer, device)
    elif TASK == "sequence":
        train_sequence_inverter(lm, tokenizer, device)
    else:
        raise ValueError("TASK must be 'token' or 'sequence'")


if __name__ == "__main__":
    main()
