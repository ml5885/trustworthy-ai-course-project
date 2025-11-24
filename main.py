import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from sipit import SipIt


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
    print(f"Computing steering vector from {len(sentences)} normal vs UPPERCASE example pairs...")

    diffs = []
    with torch.no_grad():
        for normal in sentences:
            caps = normal.upper()

            ids_norm = tok.encode(normal, return_tensors="pt").to(device)
            ids_caps = tok.encode(caps, return_tensors="pt").to(device)

            out_norm = model(input_ids=ids_norm, output_hidden_states=True)
            out_caps = model(input_ids=ids_caps, output_hidden_states=True)

            h_norm = out_norm.hidden_states[layer+1][0]
            h_caps = out_caps.hidden_states[layer+1][0]

            v_norm = h_norm[-1]
            v_caps = h_caps[-1]

            diff = (v_caps - v_norm).cpu()
            diffs.append(diff)

    diffs_tensor = torch.stack(diffs, dim=0)
    mean_diff = diffs_tensor.mean(dim=0)

    steering_vector = F.normalize(mean_diff, dim=0).to(device)
    pair_norms = diffs_tensor.norm(dim=1)
    avg_pair_norm = float(pair_norms.mean().item())

    print(f"Steering vector computed from {len(sentences)} pairs (shape: {steering_vector.shape}, avg_pair_norm: {avg_pair_norm:.4f}, normalized_norm: {steering_vector.norm().item():.4f})")

    return steering_vector, avg_pair_norm

def judge_uppercase(text):
    upper_count = sum(1 for c in text if c.isupper())
    return upper_count / len(text)

def make_hook(steer, scale):
    def hook(module, inp, out):
        sv = steer.view(1, 1, -1) * float(scale)
        if isinstance(out, tuple):
            out0 = out[0].clone()
            out0[:, -1:, :] = out0[:, -1:, :] + sv
            return (out0,) + out[1:]
        elif isinstance(out, list):
            out_list = list(out)
            out_list[0] = out_list[0].clone()
            out_list[0][:, -1:, :] = out_list[0][:, -1:, :] + sv
            return out_list
        else:
            out0 = out.clone()
            out0[:, -1:, :] = out0[:, -1:, :] + sv
            return out0
    return hook

def register_steering_hook(model, layer, steering_vector, steering_scale, verbose=False):
    blocks = None
    if hasattr(model, "layers"):
        blocks = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h

    if blocks is None:
        raise RuntimeError(f"Cannot find transformer blocks on model; cannot register hook for layer {layer}.")

    if not (0 <= layer < len(blocks)):
        raise IndexError(f"Requested layer {layer} out of range (0..{len(blocks)-1}).")

    target_block = blocks[layer]
    hook_handle = target_block.register_forward_hook(make_hook(steering_vector, steering_scale))
    if verbose: print(f"Registered steering hook on block {layer} with scale {steering_scale}")
    return hook_handle

def run_steering_experiment(model, tok, layer, device, test_prompts, steering_vector, steering_scale, verbose=False):
    hook_handle = register_steering_hook(model, layer, steering_vector, steering_scale, verbose)
    
    gen_length = 10
    sample_top_k = 50
    sample_temperature = 1.0
    
    if verbose: print("\n--- Running Steering Experiment (Generation) ---")
    
    results = []
    
    for prompt in test_prompts:
        cur_ids = tok.encode(prompt, return_tensors="pt").to(device)[0].tolist()

        generated = cur_ids.copy()
        
        for step in range(gen_length):
            ids_tensor = torch.tensor(generated, device=device).unsqueeze(0)
            
            out = model(input_ids=ids_tensor, output_hidden_states=True, use_cache=False)
            logits = out.logits

            logit = logits[0, -1]
            if sample_top_k is not None and sample_top_k > 0:
                vals, idxs = torch.topk(logit, k=min(sample_top_k, logit.size(-1)))
                probs = F.softmax(vals / float(sample_temperature), dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                next_id = int(idxs[choice].item())
            else:
                next_id = int(logit.argmax(-1).item())
            generated.append(next_id)

        gen_text = tok.decode(torch.tensor(generated), skip_special_tokens=False)
        
        full_text_len = len(tok.decode(torch.tensor(cur_ids), skip_special_tokens=False))
        new_ids = generated[len(cur_ids):]
        new_text = tok.decode(torch.tensor(new_ids), skip_special_tokens=False)
        
        score = judge_uppercase(new_text)
        results.append(score)
        
        if verbose: print(f"Steered generation: {prompt} {new_text}")
        if verbose: print(f"Uppercase Score: {score:.2f}")

    avg_score = sum(results) / len(results) if results else 0.0
    
    hook_handle.remove()
    if verbose: print(f"Removed hook from block {layer}")
    
    return avg_score

def run_recovery_experiment(model, tok, layer, device, test_prompts, steering_vector=None, steering_scale=0.0):
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
                target_h = out.hidden_states[layer+1][0, -1, :].detach().to(device)
                
                # Ensure target hidden states are the STEERED hidden states (hidden state + steering vector)
                if steering_vector is not None:
                    target_h = target_h + steering_vector * steering_scale

            (recovered_id,
             pmin, pmax, pavg,
             dmin, dmax, davg,
             gmin, gmax, gavg,
             policy_loss_per_candidate,
             dist_per_candidate,
            ) = inverter.recover_position(
                t=t,
                prefix_tokens=prefix_ids[:t],
                target_h=target_h,
                return_closest=True,
                early_stop=False,
            )
            if recovered_id is None:
                print(f"Failed to recover token at position {t+1}.")
                break
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
            
            prompt_dir = f"prompt_{t+1}"
            os.makedirs(prompt_dir, exist_ok=True)

            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.family': "serif"})

            n = len(dists)
            indices = np.arange(n)
            sc = plt.scatter(dists, p_loss, c=indices, cmap='viridis_r', alpha=0.8)
            cbar = plt.colorbar(sc)
            cbar.set_label('Candidate order (0 = first, larger = later)')

            plt.title(f"Position {t+1} - Policy Loss vs Distance")
            plt.xlabel("Distance Norm")
            plt.ylabel("Policy Gradient Loss")
            plt.grid(True)
            plt.savefig(os.path.join(prompt_dir, f"scatter_position_{t+1}.png"))
            plt.close()
            
            trials = len(gmin)
            x = np.arange(1, trials + 1)

            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.family': "serif"})

            plt.fill_between(x, gmin, gmax, color='lightblue', alpha=0.3, label='Min-Max Range')
            plt.plot(x, gmin, label='Grad Norm Min', color='blue')
            plt.plot(x, gmax, label='Grad Norm Max', color='red')
            plt.plot(x, gavg, label='Grad Norm Avg', color='green')

            plt.title(f"Position {t+1} - Gradient Norms over Trials")
            plt.xlabel("Trials")
            plt.ylabel("Gradient Norm")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(prompt_dir, f"gradnorms_position_{t+1}.png"))
            plt.close()
            
        recovered_text = tok.decode(torch.tensor(recovered_ids), skip_special_tokens=False)
        print(f"Recovered prompt using SipIT: {recovered_text}")

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
        layer=layer+1,
        step_size=0.1,
        device=device,
        inner_steps=8,
        topk=256,
        bf_batch_size=256,
        use_cosine=True,
        norm_clip=1.0,
        project_every=25,
        project_always=False,
        max_vocab_scan=10000,
        epsilon=1e-1,
        verbose=True,
    )

    random.seed(42)
    steering_vector, avg_pair_norm = compute_steering_vector(model, tok, layer, device)

    test_prompts = [
        'John said to Mary "',
        'The chef whispered, "',
        'Yesterday I told him that',
        'She asked politely, "',
        'In a letter he wrote, "'
    ]

    gen_length = 10
    scales = [5, 6, 7, 8, 9, 10]
    layers_to_test = [4, 8, 12, 16, 20]
    sample_top_k = 50
    sample_temperature = 1.0

    print(f"Model emb dim: {model.get_input_embeddings().weight.shape}")

    # pbar = tqdm(total=len(scales)*len(layers_to_test))
    # for scale in scales:
    #     scores = []
        
    #     for layer_idx in layers_to_test:
    #         score = run_steering_experiment(model, tok, layer_idx, device, test_prompts, steering_vector, scale, verbose=True)
    #         scores.append(score)
    #         pbar.update(1)

    #     plt.figure(figsize=(8, 6))
    #     plt.rcParams.update({'font.family': "serif"})
        
    #     plt.plot(layers_to_test, scores, marker='o')
    #     plt.title(f"Uppercaseness vs Layer (Steering Scale {scale})")
    #     plt.xlabel("Layer Number")
    #     plt.ylabel("All Caps Fraction")
    #     plt.grid(True)
    #     plt.xticks(layers_to_test)
    #     plt.savefig(f"layer_sweep_uppercase_{scale}.png")
    #     print(f"Saved figure to layer_sweep_uppercase_{scale}.png")

    layer = 4
    scale = 9
    run_steering_experiment(model, tok, layer, device, test_prompts, steering_vector, scale, verbose=True)
    run_recovery_experiment(model, tok, layer, device, test_prompts, steering_vector=steering_vector, steering_scale=scale)
