import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from sipit import SipIt


TEMPLATES = [
    "The {noun} is {adj}.",
    "{name} loves {noun}.",
    "I saw a {noun} at the {noun2}.",
    "This {noun} tastes {adj}.",
    "Please bring the {noun}.",
    "He wrote a {noun} yesterday.",
    "{name} will arrive soon.",
    "We visited the {noun} last {day}.",
    "{name} is friends with {name2}.",
    "Call the {noun} now."
]

ADJECTIVES = ["sunny", "delicious", "old", "new", "quiet", "loud", "bright", "dark", "happy", "sad"]
NOUNS = ["weather", "cake", "store", "dog", "cat", "park", "book", "letter", "meeting", "party"]
NAMES = ["John", "Mary", "Alice", "Bob", "Eve", "Charlie", "David", "Helen", "Irene", "Frank"]
DAYS = ["morning", "afternoon", "evening", "noon", "night"]

def compute_steering_vector(model, tok, layer, device, n_pairs=50,
                            templates=TEMPLATES, adjectives=ADJECTIVES,
                            nouns=NOUNS, names=NAMES, days=DAYS):
    print(f"Computing steering vector from {n_pairs} normal vs UPPERCASE example pairs...")

    diffs = []
    with torch.no_grad():
        for i in range(n_pairs):
            template = random.choice(templates)
            sample = template.format(
                adj=random.choice(adjectives),
                noun=random.choice(nouns),
                noun2=random.choice(nouns),
                name=random.choice(names),
                name2=random.choice(names),
                day=random.choice(days),
            )

            normal = sample
            caps = normal.upper()

            ids_norm = tok.encode(normal, return_tensors="pt").to(device)
            ids_caps = tok.encode(caps, return_tensors="pt").to(device)

            out_norm = model(input_ids=ids_norm, output_hidden_states=True)
            out_caps = model(input_ids=ids_caps, output_hidden_states=True)

            h_norm = out_norm.hidden_states[layer][0]  # (seq_len, hidden)
            h_caps = out_caps.hidden_states[layer][0]

            v_norm = h_norm[-1]
            v_caps = h_caps[-1]

            diff = (v_caps - v_norm).cpu()
            diffs.append(diff)

    if len(diffs) == 0:
        raise ValueError("No pairs were generated to compute steering vector.")

    diffs_tensor = torch.stack(diffs, dim=0)
    mean_diff = diffs_tensor.mean(dim=0)

    steering_vector = F.normalize(mean_diff, dim=0).to(device)
    pair_norms = diffs_tensor.norm(dim=1)
    avg_pair_norm = float(pair_norms.mean().item())

    print(f"Steering vector computed from {n_pairs} pairs (shape: {steering_vector.shape}, avg_pair_norm: {avg_pair_norm:.4f}, normalized_norm: {steering_vector.norm().item():.4f})")

    return steering_vector, avg_pair_norm

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # model_name = "gpt2"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    layer = 8
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model.to(device)
    inverter = SipIt(
        model,
        tok,
        layer=layer,
        step_size=0.1,
        device=device,
        inner_steps=4,
        topk=256,
        use_cosine=False,
        norm_clip=1.0,
        project_every=25,
        project_always=False,
        max_vocab_scan=5000,
        epsilon=1e-3,
        verbose=True,
    )

    random.seed(42)
    n_pairs = 50
    # steering_vector, _ = compute_steering_vector(model, tok, layer, device, n_pairs=n_pairs)

    test_prompts = [
        # 'John said to Mary "',
        'The chef whispered, "',
        # 'Yesterday I told him that',
        # 'She asked politely, "',
        # 'In a letter he wrote, "'
    ]

    gen_length = 10
    steering_scale = 15.0
    sample_top_k = 50
    sample_temperature = 1.0

    print(f"Model emb dim: {model.get_input_embeddings().weight.shape}, target layer index: {layer}")

    results = []

    # for prompt in test_prompts:
    #     print('\nPrompt:', prompt)
    #     cur_ids = tok.encode(prompt, return_tensors="pt").to(device)[0].tolist()

    #     generated = cur_ids.copy()
    #     last_hb = None

    #     transformer = getattr(model, 'transformer', None)
    #     hook_handle = None

    #     blocks = None
    #     if hasattr(model, "layers"):
    #         blocks = model.layers
    #     elif hasattr(model, "model") and hasattr(model.model, "layers"):
    #         blocks = model.model.layers
    #     elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    #         blocks = model.transformer.h

    #     if blocks is None:
    #         raise RuntimeError(f"Cannot find transformer blocks on model; cannot register hook for layer {layer}.")

    #     if not (0 <= layer < len(blocks)):
    #         raise IndexError(f"Requested layer {layer} out of range (0..{len(blocks)-1}).")

    #     target_block = blocks[layer]

    #     def make_hook(steer, scale):
    #         def hook(module, inp, out):
    #             sv = steer.view(1, 1, -1) * float(scale)
    #             out0 = out[0].clone()
    #             out0[:, -1:, :] = out0[:, -1:, :] + sv
    #             if isinstance(out, tuple):
    #                 new_out = (out0,) + tuple(out[1:])
    #             elif isinstance(out, list):
    #                 out_list = list(out)
    #                 out_list[0] = out0
    #                 new_out = out_list
    #             else:
    #                 new_out = out0
    #             return new_out
    #         return hook

    #     hook_handle = target_block.register_forward_hook(make_hook(steering_vector, steering_scale))
    #     print(f"Registered hook on block {layer}")
        
    #     for step in range(gen_length):
    #         ids_tensor = torch.tensor(generated, device=device).unsqueeze(0)
    #         inputs_embeds = model.get_input_embeddings()(ids_tensor)

    #         if hook_handle is None:
    #             if inputs_embeds.size(-1) == steering_vector.size(0):
    #                 inputs_embeds[0, -1, :] = inputs_embeds[0, -1, :] + steering_scale * steering_vector
    #             else:
    #                 proj = steering_vector[: inputs_embeds.size(-1)]
    #                 inputs_embeds[0, -1, :] = inputs_embeds[0, -1, :] + steering_scale * proj

    #         out = model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
    #         logits = out.logits

    #         logit = logits[0, -1]
    #         if sample_top_k is not None and sample_top_k > 0:
    #             vals, idxs = torch.topk(logit, k=min(sample_top_k, logit.size(-1)))
    #             probs = F.softmax(vals / float(sample_temperature), dim=-1)
    #             choice = torch.multinomial(probs, num_samples=1)
    #             next_id = int(idxs[choice].item())
    #         else:
    #             next_id = int(logit.argmax(-1).item())
    #         generated.append(next_id)

    #         last_hb = out.hidden_states[layer][0, -1, :].detach().to(device)

    #     gen_text = tok.decode(torch.tensor(generated), skip_special_tokens=False)
    #     results.append({
    #         'prompt': prompt,
    #         'gen_text': gen_text,
    #         'gen_ids': generated,
    #         'last_hb': last_hb,
    #         'recovered_id': None,
    #     })

    #     # print full steered continuation for this prompt
    #     print(f"Steered generation (strength = {steering_scale}): {gen_text}")

    #     if hook_handle is not None:
    #         hook_handle.remove()
    #         print(f"Removed hook from block {layer}")

    for prompt in test_prompts:
        print('\nPrompt:', prompt)
        
        # recover prompt from hidden states
        prefix_ids = tok.encode(prompt, return_tensors="pt").to(device)[0].tolist()
        recovered_ids = []
        policy_min_list = []
        policy_max_list = []
        policy_avg_list = []
        dist_min_list = []
        dist_max_list = []
        dist_avg_list = []
        grad_norm_min_list = []
        grad_norm_max_list = []
        grad_norm_avg_list = []
        policy_loss_per_candidate_per_token = []
        dist_per_candidate_per_token = []
        for t in range(len(prefix_ids)):
            target_h = None
            with torch.no_grad():
                ids_tensor = torch.tensor(prefix_ids[:t+1], device=device).unsqueeze(0)
                out = model(input_ids=ids_tensor, output_hidden_states=True)
                target_h = out.hidden_states[layer][0, -1, :].detach().to(device)

            (   recovered_id,
                pmin, pmax, pavg,
                dmin, dmax, davg,
                gmin, gmax, gavg,
                policy_loss_per_candidate,
                dist_per_candidate,
            ) = inverter.recover_position(
                t=t,
                prefix_tokens=prefix_ids[:t],
                target_h=target_h,
            )
            if recovered_id is None:
                print(f"Failed to recover token at position {t+1}.")
                recovered_ids.append(prefix_ids[t])
            else:
                recovered_ids.append(recovered_id)
                
            policy_loss_per_candidate_per_token.append(policy_loss_per_candidate)
            dist_per_candidate_per_token.append(dist_per_candidate)
            
            policy_min_list.append(min(pavg))
            policy_max_list.append(max(pavg))
            policy_avg_list.append(sum(pavg) / len(pavg))
            
            dist_min_list.append(min(davg))
            dist_max_list.append(max(davg))
            dist_avg_list.append(sum(davg) / len(davg))
            
            grad_norm_min_list.append(gmin)
            grad_norm_max_list.append(gmax)
            grad_norm_avg_list.append(gavg)
            
            p_loss = policy_loss_per_candidate
            dists = dist_per_candidate
            
            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.family': "serif"})

            n = len(dists)
            indices = np.arange(n)
            # use reversed viridis so index 0 -> light, last -> dark
            sc = plt.scatter(dists, p_loss, c=indices, cmap='viridis_r', alpha=0.8)
            cbar = plt.colorbar(sc)
            cbar.set_label('Candidate order (0 = first, larger = later)')

            plt.title(f"Position {t+1} - Policy Loss vs Distance")
            plt.xlabel("Distance Norm")
            plt.ylabel("Policy Gradient Loss")
            plt.grid(True)
            plt.savefig(f"scatter_position_{t+1}.png")
            plt.close()
                
            trials = len(gmin)

            x = np.arange(1, trials + 1)

            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.family': "serif"})

            # Shaded area between min and max
            plt.fill_between(x, gmin, gmax, color='lightblue', alpha=0.3, label='Min-Max Range')

            # Plot lines on top of the shading
            plt.plot(x, gmin, label='Grad Norm Min', color='blue')
            plt.plot(x, gmax, label='Grad Norm Max', color='red')
            plt.plot(x, gavg, label='Grad Norm Avg', color='green')

            plt.title(f"Position {t+1} - Gradient Norms over Trials")
            plt.xlabel("Trials")
            plt.ylabel("Gradient Norm")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"gradnorms_position_{t+1}.png")
            plt.close()
            
        recovered_text = tok.decode(torch.tensor(recovered_ids), skip_special_tokens=False)
        print(f"Recovered prompt using SipIT: {recovered_text}")
            