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
        verbose=False,
        inner_steps=3,
        topk=5,
        use_cosine=True,
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
        trials = 0
        
        pbar = tqdm(total=self.max_vocab_scan, disable=not self.verbose, desc=f"Recovering token at position {t+1}", unit="candidates")
        while trials < self.max_vocab_scan:
            v_star, e_new, ranked = self.policy_gradient(
                prefix_tokens=prefix_tokens,
                target_h=target_h,
                visited=visited,
                e_prev=e_prev,
                step_index=trials + 1,
            )
                
            for v_j in ranked:
                pred_h = self.forward_discrete(prefix_tokens, v_j)
                dist = torch.norm(pred_h - target_h, p=2)
                trials += 1

                if dist <= self.epsilon:
                    return int(v_j)
                else:
                    visited.add(int(v_j))

                if trials >= self.max_vocab_scan:
                    break

            e_prev = e_new
            pbar.update(len(ranked))
        pbar.close()

        raise RuntimeError(
            f"SIP-IT failed to verify any token at position {t+1} within {self.max_vocab_scan} proposals."
        )

    def policy_gradient(self, prefix_tokens, target_h, visited, e_prev, step_index):
        e = e_prev.detach().clone().to(self.device).requires_grad_(True)
        
        for _ in range(max(1, int(self.inner_steps))):
            pred_from_cont = self.forward_continuous(prefix_tokens, e)
            loss = 0.5 * torch.sum((pred_from_cont - target_h) ** 2)
            g = torch.autograd.grad(loss, e, retain_graph=False)[0]

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
        return v_star, e_new, ranked

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

    def nearest_token_ids(self, vec, visited, topk=5, use_cosine=True):
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
