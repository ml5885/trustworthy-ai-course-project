from typing import Optional
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import torch.nn.functional as F


class SipIt:
    def __init__(
        self,
        model,
        tokenizer,
        layer=0,
        epsilon=1e-3,
        step_size=0.1,
        device=None,
        project_every=50,
        max_vocab_scan=5000,
        bf_batch_size=256,
        verbose=False,
        inner_steps=3,
        topk=5,
        use_cosine=False,
        norm_clip=1.0,
        project_always=False,
    ):
        """SipIT class for recovering tokens using target hidden states.
        
        Args:
            layer: Layer index to target for hidden states.
            epsilon: Acceptable distance threshold for recovery.
            step_size: Step size for gradient updates.
            project_every: Frequency of projection to nearest token.
            max_vocab_scan: Maximum number of token proposals to try.
            inner_steps: Number of inner optimization steps per proposal.
            topk: Number of nearest tokens to consider per proposal.
            use_cosine: If true use cosine similarity for nearest token search.
                        Otherwise, use Euclidean distance.
            norm_clip: If set, clip gradient norms to this value.
            project_always: Whether to always project to nearest token after inner steps.
        """
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tok = tokenizer

        self.layer = layer
        self.epsilon = epsilon
        self.step_size = step_size
        self.project_every = project_every
        self.max_vocab_scan = max_vocab_scan
        self.verbose = verbose
        self.inner_steps = inner_steps
        self.topk = topk
        self.use_cosine = use_cosine
        self.norm_clip = norm_clip
        self.project_always = project_always
        self.bf_batch_size = bf_batch_size

        self.device = device or (
            next(self.model.parameters()).device if any(True for _ in self.model.parameters()) else "cpu"
        )

        self.emb = self.model.get_input_embeddings().weight
        self.emb = self.emb.to(self.device)
        self._emb_norm = None

        self.vocab_size, self.hidden_size = self.emb.shape

        if self.max_vocab_scan is None:
            self.max_vocab_scan = self.vocab_size

    def recover_position(self, t, prefix_tokens, target_h):
        visited = set()
        
        if len(prefix_tokens) > 0:
            e_prev = self.emb[prefix_tokens[-1]].detach().clone()
        else:
            e_prev = self.emb.mean(dim=0).detach().clone()
        
        e_prev = e_prev.to(self.device)
        # ensure target hidden state is on the same device
        target_h = target_h.to(self.device)
        trials = 0
        
        policy_min = []
        policy_max = []
        policy_avg = []
        dist_min = []
        dist_max = []
        dist_avg = []
        grad_norm_min = []
        grad_norm_max = []
        grad_norm_avg = []

        policy_loss_per_candidate = []
        dist_per_candidate = []
        
        pbar = tqdm(total=self.max_vocab_scan, disable=not self.verbose, desc=f"Recovering token at position {t+1}", unit="candidates")
        while trials < self.max_vocab_scan:
            v_star, e_new, ranked, losses, grads = self.policy_gradient(
                prefix_tokens=prefix_tokens,
                target_h=target_h,
                visited=visited,
                e_prev=e_prev,
                step_index=trials + 1,
            )
                
            policy_min.append(min(losses))
            policy_max.append(max(losses))
            policy_avg.append(sum(losses) / len(losses))
            policy_val = losses[-1]
            
            grad_norm_min.append(min(grads))
            grad_norm_max.append(max(grads))
            grad_norm_avg.append(sum(grads) / len(grads))
            
            # Evaluate all ranked candidate tokens in a single batched model call
            n_ranked = len(ranked)
            if n_ranked > 0:
                if len(prefix_tokens) > 0:
                    prefix_ids = torch.tensor(prefix_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
                    prefix_rep = prefix_ids.repeat(n_ranked, 1)
                    cand_ids = torch.tensor(ranked, device=self.device, dtype=torch.long).unsqueeze(1)
                    ids = torch.cat([prefix_rep, cand_ids], dim=1)
                else:
                    ids = torch.tensor(ranked, device=self.device, dtype=torch.long).unsqueeze(1)

                out = self.model(
                    input_ids=ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                h_layer = out.hidden_states[self.layer]
                preds = h_layer[:, -1, :]

                diffs = preds - target_h.unsqueeze(0)
                dists_tensor = torch.norm(diffs, p=2, dim=1)
                dists = dists_tensor.cpu().tolist()

                # process distances in the original ranked order, preserving early-return behavior
                hit_index = None
                for idx, dist_val in enumerate(dists):
                    policy_loss_per_candidate.append(float(policy_val))
                    dist_per_candidate.append(float(dist_val))
                    trials += 1
                    if dist_val <= self.epsilon and hit_index is None:
                        hit_index = idx
                        break
                    else:
                        visited.add(int(ranked[idx]))

                    if trials >= self.max_vocab_scan:
                        break

                if hit_index is not None:
                    # compute stats only over the scanned candidates up to and including the hit
                    prefix_dists = dists[: hit_index + 1]
                    dist_min.append(min(prefix_dists))
                    dist_max.append(max(prefix_dists))
                    dist_avg.append(sum(prefix_dists) / len(prefix_dists))
                    v_j = ranked[hit_index]
                    return (
                        int(v_j),
                        policy_min,
                        policy_max,
                        policy_avg,
                        dist_min,
                        dist_max,
                        dist_avg,
                        grad_norm_min,
                        grad_norm_max,
                        grad_norm_avg,
                        policy_loss_per_candidate,
                        dist_per_candidate,
                    )
                else:
                    # no hit in this batch: update summary stats for the whole batch
                    if dists:
                        dist_min.append(min(dists))
                        dist_max.append(max(dists))
                        dist_avg.append(sum(dists) / len(dists))

            if dists:
                dist_min.append(min(dists))
                dist_max.append(max(dists))
                dist_avg.append(sum(dists) / len(dists))
            
            e_prev = e_new
            pbar.update(len(ranked))
        # pbar.close()
        
        # If we exhausted the guided proposals without finding a token within epsilon,
        # do a brute-force scan over remaining vocabulary tokens in batches to speed up inference.
        remaining = [i for i in range(self.vocab_size) if i not in visited]
        if remaining:
            # process remaining tokens in chunks
            pbar = tqdm(total=len(remaining), disable=not self.verbose, desc=f"Brute-force scanning remaining tokens at position {t+1}", unit="candidates")
            for i in range(0, len(remaining), max(1, int(self.bf_batch_size))):
                chunk = remaining[i : i + int(self.bf_batch_size)]
                batch_size = len(chunk)

                # build input_ids for the batch: each row = prefix_tokens + candidate_token
                if len(prefix_tokens) > 0:
                    prefix_ids = torch.tensor(prefix_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
                    prefix_rep = prefix_ids.repeat(batch_size, 1)
                    cand_ids = torch.tensor(chunk, device=self.device, dtype=torch.long).unsqueeze(1)
                    ids = torch.cat([prefix_rep, cand_ids], dim=1)
                else:
                    # only the candidate token per row
                    ids = torch.tensor(chunk, device=self.device, dtype=torch.long).unsqueeze(1)

                # run the model once for the whole chunk
                out = self.model(
                    input_ids=ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                h_layer = out.hidden_states[self.layer]
                # h_layer shape: (batch_size, seq_len, hidden_size); take last token hidden
                preds = h_layer[:, -1, :]

                # compute L2 distances in a vectorized manner
                diffs = preds - target_h.unsqueeze(0)
                dists = torch.norm(diffs, p=2, dim=1).cpu().tolist()

                # record metrics for this chunk
                policy_loss_per_candidate.extend([float('nan')] * batch_size)
                dist_per_candidate.extend([float(d) for d in dists])

                # update distance summaries for this chunk
                if dists:
                    dist_min.append(min(dists))
                    dist_max.append(max(dists))
                    dist_avg.append(sum(dists) / len(dists))

                # check if any candidate in this chunk meets epsilon
                min_idx = int(torch.argmin(torch.tensor(dists)).item())
                best_dist = dists[min_idx]
                best_token = int(chunk[min_idx])
                if best_dist <= self.epsilon:
                    return (
                        int(best_token),
                        policy_min,
                        policy_max,
                        policy_avg,
                        dist_min,
                        dist_max,
                        dist_avg,
                        grad_norm_min,
                        grad_norm_max,
                        grad_norm_avg,
                        policy_loss_per_candidate,
                        dist_per_candidate,
                    )
                pbar.update(batch_size)
            pbar.close()

        # If still not found, return None with accumulated metrics
        return (
            None,
            policy_min,
            policy_max,
            policy_avg,
            dist_min,
            dist_max,
            dist_avg,
            grad_norm_min,
            grad_norm_max,
            grad_norm_avg,
            policy_loss_per_candidate,
            dist_per_candidate,
        )
        # raise RuntimeError(
        #     f"SIP-IT failed to verify any token at position {t+1} within {self.max_vocab_scan} proposals."
        # )

    def policy_gradient(self, prefix_tokens, target_h, visited, e_prev, step_index):
        e = e_prev.detach().clone().to(self.device).requires_grad_(True)
        
        losses = []
        grads = []
        for _ in range(max(1, int(self.inner_steps))):
            pred_from_cont = self.forward_continuous(prefix_tokens, e)
            loss = 0.5 * torch.sum((pred_from_cont - target_h) ** 2)
            losses.append(loss.item())
            g = torch.autograd.grad(loss, e, retain_graph=False)[0]
            grads.append(g.norm(p=2).item())

            if self.norm_clip is not None:
                g_norm = g.norm(p=2) + 1e-12
                max_n = float(self.norm_clip)
                if g_norm > max_n:
                    g = g * (max_n / g_norm)

            e = (e - self.step_size * g).detach().requires_grad_(True)

        e_new = e.detach()

        project_now = (self.project_every and step_index % self.project_every == 0) or self.project_always
        if project_now:
            nearest_id = self.nearest_token_id(e_new, visited, use_cosine=self.use_cosine)
            e_new = self.emb[nearest_id].detach().clone()

        ranked = self.nearest_token_ids(e_new, visited, topk=max(1, int(self.topk)), use_cosine=self.use_cosine)
        v_star = int(ranked[0])
        return v_star, e_new, ranked, losses, grads

    def nearest_token_id(self, vec, visited, use_cosine=True):
        if use_cosine:
            if self._emb_norm is None or self._emb_norm.device != self.device or self._emb_norm.shape != self.emb.shape:
                self._emb_norm = F.normalize(self.emb, dim=1)
            v = F.normalize(vec, dim=0)
            sims = torch.matmul(self._emb_norm, v) 
            scores = -sims
        else:
            dists = torch.sum((self.emb - vec.unsqueeze(0)) ** 2, dim=1)
            scores = dists

        if visited:
            scores[list(visited)] = float("inf")
        return int(torch.argmin(scores).item())

    def nearest_token_ids(self, vec, visited, topk=5, use_cosine=False):
        topk = max(1, int(topk))
        if use_cosine:
            if self._emb_norm is None or self._emb_norm.device != self.device or self._emb_norm.shape != self.emb.shape:
                self._emb_norm = F.normalize(self.emb, dim=1)
            v = F.normalize(vec, dim=0)
            sims = torch.matmul(self._emb_norm, v)
            scores = -sims
        else:
            dists = torch.sum((self.emb - vec.unsqueeze(0)) ** 2, dim=1)
            scores = dists

        if visited:
            scores[list(visited)] = float("inf")

        vals, idxs = torch.topk(-scores, k=min(topk, self.vocab_size - len(visited)), largest=True)
        idxs = idxs.tolist()
        return idxs

    def forward_discrete(self, prefix_tokens, candidate_token):
        ids = torch.tensor(prefix_tokens + [candidate_token], device=self.device).unsqueeze(0)
        out = self.model(
            input_ids=ids,
            output_hidden_states=True,
            use_cache=False,
        )
        h_layer = out.hidden_states[self.layer]
        return h_layer[0, -1, :]

    def forward_continuous(self, prefix_tokens, e):
        if len(prefix_tokens) > 0:
            prefix_ids = torch.tensor(prefix_tokens, device=self.device).unsqueeze(0)
            prefix_embeds = self.model.get_input_embeddings()(prefix_ids)
            seq_len = prefix_embeds.size(1) + 1
        else:
            prefix_embeds = torch.empty((1, 0, self.hidden_size), device=self.device)
            seq_len = 1

        e = e.view(1, 1, -1)
        inputs_embeds = torch.cat([prefix_embeds, e], dim=1)

        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)

        out = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        h_layer = out.hidden_states[self.layer]
        return h_layer[0, -1, :]