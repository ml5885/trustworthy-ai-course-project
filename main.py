import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUFFIX_LEN = 20
REPLACEMENT_LEN = 20
GCG_STEPS = 200
BATCH_SIZE = 512
TOP_K = 64

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

def compute_steering_vector(model, tok, layer, device, sentences=STEERING_SENTENCES):
    print(f"Computing steering vector from {len(sentences)} pairs...")
    diffs = []
    with torch.no_grad():
        for normal in sentences:
            caps = normal.upper()
            ids_norm = tok.encode(normal, return_tensors="pt").to(device)
            ids_caps = tok.encode(caps, return_tensors="pt").to(device)

            out_norm = model(input_ids=ids_norm, output_hidden_states=True, use_cache=False)
            out_caps = model(input_ids=ids_caps, output_hidden_states=True, use_cache=False)

            h_norm = out_norm.hidden_states[layer+1][0][-1]
            h_caps = out_caps.hidden_states[layer+1][0][-1]

            diff = (h_caps - h_norm).cpu()
            diffs.append(diff)

    diffs_tensor = torch.stack(diffs, dim=0)
    mean_diff = diffs_tensor.mean(dim=0)
    steering_vector = F.normalize(mean_diff, dim=0).to(device)
    print(f"Vector computed. Norm: {steering_vector.norm().item():.4f}")
    return steering_vector

def make_hook(steer, scale):
    def hook(module, inp, out):
        sv = steer.view(1, 1, -1) * float(scale)
        if isinstance(out, tuple):
            out[0][:, -1:, :].add_(sv.to(out[0].dtype))
            return out
        elif hasattr(out, "last_hidden_state"):
            out.last_hidden_state[:, -1:, :].add_(sv.to(out.last_hidden_state.dtype))
            return out
        else:
            out[:, -1:, :].add_(sv.to(out.dtype))
            return out
    return hook

def register_steering_hook(model, layer, steering_vector, steering_scale):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "layers"):
        blocks = model.layers
    else:
        raise ValueError("Could not locate transformer blocks.")

    target_block = blocks[layer]
    handle = target_block.register_forward_hook(make_hook(steering_vector, steering_scale))
    return handle

def judge_uppercase(text):
    if not text:
        return 0.0
    return sum(1 for c in text if c.isupper()) / len(text)

def run_generation_check(model, tok, prompt, device, description):
    if len(prompt.strip()) == 0:
        print(f"[{description}] - Empty Prompt (Skipping)")
        return {"description": description, "prompt": prompt, "gen": "", "score": 0.0}

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=15, do_sample=False, use_cache=False)

    full_text = tok.decode(output[0], skip_special_tokens=False)
    new_text = full_text[len(prompt):]
    score = judge_uppercase(new_text)

    print(f"[{description}]")
    print(f"Prompt: {prompt!r}")
    print(f"Gen:    {new_text.strip()}")
    print(f"Score:  {score:.2f}\n")
    return {"description": description, "prompt": prompt, "gen": new_text.strip(), "score": score}

class VanillaGCG:
    def __init__(self, model, tokenizer, layer_idx, device, mode="suffix"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.embeddings = model.get_input_embeddings()
        self.mode = mode

    def get_final_hidden(self, input_ids, use_grad=False):
        if use_grad:
            inputs_embeds = self.embeddings(input_ids)
            inputs_embeds.retain_grad()
            outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
            return outputs.hidden_states[self.layer_idx][:, -1, :], inputs_embeds
        else:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
                return outputs.hidden_states[self.layer_idx][:, -1, :]

    def run(self, base_prompt, target_hidden_state):
        print(f"Starting Vanilla GCG (Mode: {self.mode})...")
        vocab_size = self.embeddings.weight.shape[0]

        if self.mode == "suffix":
            fixed_ids = self.tokenizer.encode(base_prompt, add_special_tokens=False, return_tensors="pt").to(self.device)[0]
            optim_len = SUFFIX_LEN
            optim_ids = torch.randint(0, vocab_size, (optim_len,), device=self.device)
        else:
            fixed_ids = torch.tensor([], dtype=torch.long, device=self.device)
            optim_len = REPLACEMENT_LEN
            optim_ids = torch.randint(0, vocab_size, (optim_len,), device=self.device)

        best_optim_ids = optim_ids.clone()
        best_loss = float('inf')

        with torch.no_grad():
            curr_input = torch.cat([fixed_ids, optim_ids]).unsqueeze(0)
            curr_h = self.get_final_hidden(curr_input)
            initial_loss = torch.norm(curr_h - target_hidden_state, p=2).item()
            print(f"Initial L2 Loss: {initial_loss:.4f}")
            best_loss = initial_loss

        loss_history = [best_loss]

        pbar = tqdm(range(GCG_STEPS), desc="Optimizing")
        for _ in pbar:
            current_input_ids = torch.cat([fixed_ids, optim_ids]).unsqueeze(0)
            current_hidden, inputs_embeds = self.get_final_hidden(current_input_ids, use_grad=True)
            loss = torch.norm(current_hidden - target_hidden_state, p=2)
            loss.backward()
            grad_slice = inputs_embeds.grad[0, -optim_len:, :]
            with torch.no_grad():
                grad_on_vocab = torch.matmul(grad_slice, self.embeddings.weight.T)
                top_indices = torch.topk(-grad_on_vocab, TOP_K, dim=1).indices

            new_candidate_ids = optim_ids.repeat(BATCH_SIZE, 1)
            pos_indices = torch.randint(0, optim_len, (BATCH_SIZE,), device=self.device)
            k_indices = torch.randint(0, TOP_K, (BATCH_SIZE,), device=self.device)
            vocab_indices = top_indices[pos_indices, k_indices]
            new_candidate_ids[torch.arange(BATCH_SIZE), pos_indices] = vocab_indices
            new_candidate_ids[0] = best_optim_ids

            if len(fixed_ids) > 0:
                fixed_expanded = fixed_ids.repeat(BATCH_SIZE, 1)
                batch_inputs = torch.cat([fixed_expanded, new_candidate_ids], dim=1)
            else:
                batch_inputs = new_candidate_ids

            mini_batch_size = 64
            losses = []
            with torch.no_grad():
                for i in range(0, BATCH_SIZE, mini_batch_size):
                    batch_slice = batch_inputs[i:i+mini_batch_size]
                    out = self.model(input_ids=batch_slice, output_hidden_states=True, use_cache=False)
                    batch_h = out.hidden_states[self.layer_idx][:, -1, :]
                    target_expanded = target_hidden_state.expand(batch_h.size(0), -1)
                    batch_loss = torch.norm(batch_h - target_expanded, p=2, dim=1)
                    losses.append(batch_loss)
                losses = torch.cat(losses)
                min_loss, min_idx = torch.min(losses, dim=0)
                if min_loss.item() < best_loss:
                    best_loss = min_loss.item()
                    best_optim_ids = new_candidate_ids[min_idx]
                    optim_ids = best_optim_ids

            loss_history.append(best_loss)
            pbar.set_postfix({"L2 Loss": f"{best_loss:.4f}"})

        return self.tokenizer.decode(best_optim_ids), best_loss, loss_history

def plot_gcg_loss(histories, labels, save_path="results/gcg_loss.png", epsilon=0.1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    for history, label in zip(histories, labels):
        ax.plot(range(len(history)), history, linewidth=2, label=label)

    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1, label=f"SipIt threshold ($\\epsilon={epsilon}$)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=8, help="Transformer block index for steering")
    parser.add_argument("--strength", type=float, default=9.0)
    parser.add_argument("--mode", default="both", choices=["suffix", "replacement", "both"])
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = {"model": args.model, "layer": args.layer, "strength": args.strength}

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model.to(DEVICE)
    model.eval()

    steering_vector = compute_steering_vector(model, tok, args.layer, DEVICE)
    test_prompt = 'John said to Mary "'

    print(f"\n--- Computing Target State (Strength {args.strength}) ---")
    hook = register_steering_hook(model, args.layer, steering_vector, args.strength)
    target_input = tok(test_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**target_input, output_hidden_states=True, use_cache=False)
        target_state = out.hidden_states[args.layer + 1][0, -1, :].detach().clone()
    hook.remove()
    print(f"Target Norm: {target_state.norm().item():.4f}")

    with torch.no_grad():
        out_unsteered = model(**target_input, output_hidden_states=True, use_cache=False)
        unsteered_state = out_unsteered.hidden_states[args.layer + 1][0, -1, :].detach()
    dist = torch.norm(target_state - unsteered_state).item()
    print(f"Sanity Check (Target - Unsteered L2): {dist:.4f}")
    if dist < 0.1:
        print("WARNING: Steering hook didn't seem to apply.")
    results["steering_l2"] = dist

    modes = [args.mode] if args.mode != "both" else ["suffix", "replacement"]
    all_histories = []
    all_labels = []
    label_map = {"suffix": "Suffix (Random Init)", "replacement": "Replacement (Prompt Init)"}

    for mode in modes:
        gcg = VanillaGCG(model, tok, args.layer + 1, DEVICE, mode=mode)
        optimized_string, final_loss, loss_history = gcg.run(test_prompt, target_state)
        all_histories.append(loss_history)
        all_labels.append(label_map[mode])

        full_prompt = (test_prompt + optimized_string) if mode == "suffix" else optimized_string

        print("\n" + "=" * 30)
        print(f"Mode: {mode}")
        print(f"Final L2 Loss: {final_loss:.4f}")
        print(f"Optimized String: {optimized_string}")
        print(f"Full Input: {full_prompt}")
        print("=" * 30 + "\n")

        gen_results = []
        print("--- Verification: Generations ---")
        gen_results.append(run_generation_check(model, tok, test_prompt, DEVICE, "Vanilla (Unsteered)"))
        hook = register_steering_hook(model, args.layer, steering_vector, args.strength)
        gen_results.append(run_generation_check(model, tok, test_prompt, DEVICE, f"Steered (Vector {args.strength})"))
        hook.remove()
        gen_results.append(run_generation_check(model, tok, full_prompt, DEVICE, f"Adversarial ({mode})"))

        results[mode] = {
            "optimized_string": optimized_string,
            "full_prompt": full_prompt,
            "final_loss": final_loss,
            "loss_history": loss_history,
            "generations": gen_results,
        }

    if all_histories:
        plot_gcg_loss(all_histories, all_labels, save_path=os.path.join(args.outdir, "gcg_loss.png"))

    with open(os.path.join(args.outdir, "gcg_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {os.path.join(args.outdir, 'gcg_results.json')}")