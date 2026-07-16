"""CooperatingExperts: the wrapper that ties multiple experts together.

Key ideas implemented here
--------------------------
1. **Switch tokens.** Every expert's vocab contains one `<switch:NAME>` token
   per expert. When an expert emits a switch token, control (and the running
   hidden state) is handed to the target expert.

2. **Shared embedding space.** Each expert has lightweight linear projections
   `to_shared` (d_model -> shared_dim) and `from_shared` (shared_dim -> d_model).
   At a switch boundary the *current* expert's last hidden state is projected
   into the shared space `z`, then the *next* expert projects `z` back into its
   own hidden space and continues generating from there.

3. **Joint fine-tuning.** During the joint phase we construct "hand-off"
   sequences: the first half is tokenized by expert A, the second half by
   expert B. We run A, take its final hidden state, project to shared space,
   project back into B's space, and feed that as the initial hidden state for
   B. The loss is B's LM loss on the second half *plus* a small alignment loss
   that encourages the round-trip A->shared->A to be close to identity
   (representation alignment regularizer from the model-stitching literature).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, ExpertConfig, SharedSpaceConfig
from model import Expert
from tokenizer import ExpertTokenizer


class CooperatingExperts(nn.Module):
    """Container for all experts + routing logic."""

    def __init__(
        self,
        config: Config,
        tokenizers: Dict[str, ExpertTokenizer],
    ):
        super().__init__()
        self.config = config
        self.expert_names: List[str] = list(config.experts.keys())
        self.tokenizers = tokenizers

        # Build one Expert per config. Vocab size comes from the tokenizer
        # (BPE merges + special tokens).
        self.experts = nn.ModuleDict()
        for name, ecfg in config.experts.items():
            vs = tokenizers[name].vocab_size
            self.experts[name] = Expert(ecfg, config.shared, vs)

    # ------------------------------------------------------------------ #
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def expert(self, name: str) -> Expert:
        return self.experts[name]  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    def _cross_attn_enabled(self) -> bool:
        """Whether the CALM-style cross-attention bridge is turned on."""
        return bool(self.config.shared.cross_attn)

    def _carry_through_shared(
        self,
        src_expert: Expert,
        h_src: torch.Tensor,
        dst_expert: Expert,
        detach: bool = False,
    ) -> torch.Tensor:
        """Project carried hidden states from one expert into another's space.

        h_src: [B, k, d_src] hidden states of the sending expert (its own
        d_model). Returns [B, k, d_dst] -- the carried states in the
        receiving expert's d_model, ready to be used either as a seed
        (encode_with_seed) or as cross-attention memory (encode_with_cross_attn).
        """
        z = src_expert.to_shared_space(h_src)        # [B, k, shared_dim]
        out = dst_expert.from_shared_space(z)        # [B, k, d_dst]
        return out.detach() if detach else out

    def _handoff_query_ids(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Prepend a single pad "hand-off query" position to `ids`.

        This is the unified hand-off convention used by joint_loss,
        mixed_loss and generate: after a switch, a leading pad position acts
        as the query that predicts the FIRST destination token, then each
        real token predicts its successor. Using the same convention in every
        path guarantees training matches inference.
        """
        handoff = torch.full(
            (ids.size(0), 1), self.tokenizers[name].pad_id,
            dtype=ids.dtype, device=ids.device,
        )
        return torch.cat([handoff, ids], dim=1)

    def _encode_receiver(
        self,
        name: str,
        expert: Expert,
        ids: torch.Tensor,
        carried: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a segment that RECEIVES a hand-off (carried != None) or not.

        Returns (logits, targets) where logits predict `targets` position-for-
        position, under the SAME convention for both bridge modes:

          - no carry: standard causal LM (logits[:, :-1] predict ids[:, 1:]).
          - with carry: a leading pad hand-off query predicts the first token,
            so logits predict all of `ids`.

        For the seed-prepend path the carried states are prepended as virtual
        positions; the hand-off query then sits right after them and its
        output is the first prediction, exactly mirroring generation.
        """
        if carried is None:
            h = (expert.encode_with_cross_attn(ids, None)
                 if self._cross_attn_enabled() else expert.encode(ids))
            logits = expert.logits_from_hidden(h[:, :-1, :])
            return logits, ids[:, 1:], h

        query_ids = self._handoff_query_ids(name, ids)
        if self._cross_attn_enabled():
            h = expert.encode_with_cross_attn(query_ids, carried)  # [B, 1+T, d]
            logits = expert.logits_from_hidden(h[:, :-1, :])       # predict ids
            return logits, ids, h
        # Seed-prepend path: [B, K + (1+T), d]. The K seed outputs are
        # discarded; the hand-off-query output predicts the first token.
        k = carried.size(1)
        h = expert.encode_with_seed(query_ids, carried)
        logits = expert.logits_from_hidden(h[:, k:k + query_ids.size(1) - 1, :])
        return logits, ids, h

    # ------------------------------------------------------------------ #
    # Pre-training loss for a single expert (standard next-token LM).
    # ------------------------------------------------------------------ #
    def pretrain_loss(
        self, name: str, ids: torch.Tensor
    ) -> torch.Tensor:
        """Standard causal LM loss for one expert.

        ids: [B, T] of token ids in this expert's vocab.
        """
        exp = self.expert(name)
        logits, _ = exp(ids[:, :-1])
        targets = ids[:, 1:]
        pad_id = self.tokenizers[name].pad_id
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )
        return loss

    # ------------------------------------------------------------------ #
    # Joint "hand-off" loss: A produces a hidden state, it is carried through
    # the shared space, B continues. Plus an alignment regularizer.
    # ------------------------------------------------------------------ #
    def joint_loss(
        self,
        name_a: str,
        ids_a: torch.Tensor,
        name_b: str,
        ids_b: torch.Tensor,
        align_weight: float = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the joint hand-off loss between expert A and expert B.

        ids_a: [B, Ta] prefix tokens in A's vocab.
        ids_b: [B, Tb] continuation tokens in B's vocab.

        Steps:
          1. Run A on ids_a, take the last hidden state h_a  [B, d_a].
          2. Project to shared space:  z = A.to_shared(h_a)  [B, shared_dim].
          3. Project into B's space:   h_b0 = B.from_shared(z) [B, d_b].
          4. Run B on ids_b, but *prepend* h_b0 as the initial state by
             concatenating it as a virtual "position 0" hidden state.
          5. LM loss = B's next-token loss on ids_b (shifted), conditioned on
             the carried-over state.
          6. Alignment regularizer: ||A.from_shared(A.to_shared(h_a)) - h_a||^2
             encourages the projections to be (approximately) invertible, which
             is the key assumption behind model stitching.
        """
        exp_a = self.expert(name_a)
        exp_b = self.expert(name_b)
        if align_weight is None:
            align_weight = self.config.train.align_weight
        K = self.config.shared.bridge_len

        # 1. Encode prefix with A, take the LAST k hidden states (the tokens
        #    right at the hand-off boundary). k = bridge_len, clamped to Ta.
        h_full = exp_a.encode(ids_a)              # [B, Ta, d_a]
        k = min(K, h_full.size(1))
        seed_src = h_full[:, -k:, :]              # [B, k, d_a]

        # 2-3. Carry the k boundary states through the shared bottleneck into
        #      B's hidden space.
        z_a = exp_a.to_shared_space(seed_src)     # [B, k, shared_dim]
        seed_b = exp_b.from_shared_space(z_a)     # [B, k, d_b]

        # 4-5. Run B under the UNIFIED hand-off convention (a leading pad
        #      query predicts the first token). _encode_receiver handles both
        #      bridge modes identically to generation.
        logits_b, targets_b, h_b = self._encode_receiver(
            name_b, exp_b, ids_b, seed_b,
        )
        pad_b = self.tokenizers[name_b].pad_id
        lm_loss = F.cross_entropy(
            logits_b.reshape(-1, logits_b.size(-1)),
            targets_b.reshape(-1),
            ignore_index=pad_b,
        )

        # 6a. Round-trip alignment: encourage each expert's own
        #     (from_shared . to_shared) to be close to identity, so the
        #     projections are (approximately) invertible.
        ref_a = seed_src.detach()
        # B's own hidden over its first k real tokens (a stable reference that
        # does not depend on the hand-off path indexing).
        ref_b_hidden = exp_b.encode(ids_b)[:, :k, :].detach()
        align_rt = (
            F.mse_loss(exp_a.from_shared_space(exp_a.to_shared_space(ref_a)), ref_a)
            + F.mse_loss(
                exp_b.from_shared_space(exp_b.to_shared_space(ref_b_hidden)),
                ref_b_hidden,
            )
        )

        # 6b. Cross-expert alignment: the whole point of a SHARED space is that
        #     A's boundary states and B's continuation states land in the same
        #     region of the bottleneck. Round-trip identity alone allows each
        #     expert to occupy a disjoint subspace. We therefore pull A's
        #     boundary shared code toward B's continuation shared code (both
        #     L2-normalized so this aligns direction, not magnitude).
        z_b = exp_b.to_shared_space(ref_b_hidden)          # [B, k, shared_dim]
        z_a_n = F.normalize(z_a.mean(dim=1), dim=-1)       # [B, shared_dim]
        z_b_n = F.normalize(z_b.mean(dim=1), dim=-1)       # [B, shared_dim]
        align_cross = (1.0 - (z_a_n * z_b_n).sum(dim=-1)).mean()

        align = align_rt + align_cross

        total = lm_loss + align_weight * align
        info = {
            "lm": lm_loss.item(),
            "align": align.item(),
            "total": total.item(),
        }
        return total, info

    # ------------------------------------------------------------------ #
    # Interleaved "mixed" loss: train BOTH experts end-to-end on whole
    # sessions where switch tokens have been inserted at real code<->text
    # boundaries. Each example is a list of (expert_name, ids) segments.
    # We carry the last hidden state through the shared space from one
    # segment to the next, exactly like generation, but compute a real LM
    # loss over every token (including the switch tokens).
    # ------------------------------------------------------------------ #
    def mixed_loss(
        self,
        segments: List[Tuple[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, dict]:
        """LM loss over an interleaved, switch-token-annotated session.

        segments: list of (expert_name, ids[B,T]) — one segment per expert
                  turn. Switch tokens are real targets at the end of a segment
                  and condition the next expert via the carried hidden state.
        """
        device = self.device()
        # Accumulate per-segment losses in a list and stack at the end.
        # (Initializing total_loss as a leaf tensor with requires_grad=True
        # and then reassigning via `+` produced a spurious graph node and
        # fragile gradient flow.)
        seg_losses: List[torch.Tensor] = []
        carried: Optional[torch.Tensor] = None  # [B, K, d] carried seed states
        K = self.config.shared.bridge_len

        for seg_idx, (name, ids) in enumerate(segments):
            exp = self.expert(name)
            ids = ids.to(device)
            # Ensure a batch dimension: [T] -> [1, T].
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            B, T = ids.shape
            if T > exp.cfg.max_seq_len:
                ids = ids[:, -exp.cfg.max_seq_len:]
                T = ids.size(1)

            # Unified hand-off convention for BOTH bridge modes (see
            # _encode_receiver): a leading pad query predicts the first token
            # after a switch, exactly like generation.
            logits, targets, h = self._encode_receiver(name, exp, ids, carried)

            pad = self.tokenizers[name].pad_id

            # Down-weight switch tokens per-TOKEN (not per-vocab-class). A
            # per-class `weight=` also rescales the overall loss magnitude,
            # making it non-comparable to the pretrain/joint LM loss. Instead
            # we compute an unreduced CE and scale only the positions whose
            # target is a switch token.
            sw_w = self.config.train.switch_loss_weight
            switch_ids = {self.tokenizers[name].switch_id(_n)
                          for _n in self.expert_names}
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            per_tok = F.cross_entropy(
                flat_logits, flat_targets, ignore_index=pad, reduction="none",
            )
            valid = flat_targets != pad
            weights = torch.ones_like(per_tok)
            is_switch = torch.zeros_like(flat_targets, dtype=torch.bool)
            for sid in switch_ids:
                is_switch |= flat_targets == sid
            weights[is_switch] = sw_w
            weights = weights * valid.float()
            denom = weights.sum().clamp_min(1.0)
            seg_loss = (per_tok * weights).sum() / denom
            seg_losses.append(seg_loss)

            # Carry the LAST k hidden states of this segment through the shared
            # space to the next expert. We DETACH the carried states so
            # gradients do not flow back through the previous expert's whole
            # transformer via from_shared -- this (a) bounds memory for long
            # batch=1 sessions, and (b) matches generation, where the carried
            # state is produced under torch.no_grad().
            next_name = segments[seg_idx + 1][0] if seg_idx + 1 < len(segments) else None
            if next_name is not None:
                k_next = min(K, h.size(1))
                carried = self._carry_through_shared(
                    exp, h[:, -k_next:, :], self.expert(next_name), detach=True,
                )
            else:
                carried = None

        if not seg_losses:
            # Degenerate session with no computable segments.
            avg = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            avg = torch.stack(seg_losses).mean()
        info = {"loss": avg.item(), "n_segs": len(seg_losses)}
        return avg, info

    # ------------------------------------------------------------------ #
    # Generation with switching.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        start_expert: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        max_switches: int = None,
    ) -> str:
        """Greedy/sample generation with live expert switching.

        Stops on EOS or when max_switches expert transitions are exhausted.
        max_switches defaults to config.gen.max_switches (= 4).
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.gen.max_new_tokens
        if temperature is None:
            temperature = self.config.gen.temperature
        if top_k is None:
            top_k = self.config.gen.top_k
        if max_switches is None:
            max_switches = self.config.gen.max_switches

        device = self.device()
        tok_a = self.tokenizers[start_expert]
        active = start_expert

        # Encode the prompt in the starting expert's vocab.
        ids = tok_a.tokenizer.encode(prompt).ids
        if len(ids) == 0:
            ids = [tok_a.pad_id]
        ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Carried seed states (None until a switch happens): [B, K, d].
        carried: Optional[torch.Tensor] = None
        switch_count: int = 0
        K = self.config.shared.bridge_len

        out_tokens: List[Tuple[str, int]] = []  # (expert, token_id)

        for _ in range(max_new_tokens):
            exp = self.expert(active)
            tok = self.tokenizers[active]
            T = ids.size(1)
            if T > exp.cfg.max_seq_len:
                ids = ids[:, -exp.cfg.max_seq_len:]

            # Just after a switch the sequence is empty: seed it with a single
            # pad "hand-off query" position, matching the training convention
            # in _encode_receiver (the query predicts the first destination
            # token). This is done for BOTH bridge modes.
            if ids.size(1) == 0:
                ids = torch.tensor([[tok.pad_id]], dtype=torch.long,
                                   device=device)

            # Encode the current sequence ONCE, seeded by the carried states
            # (if any). We keep the full last-layer hidden `h` so that, on a
            # switch, we can reuse its tail instead of re-encoding.
            if self._cross_attn_enabled():
                # CALM path: cross-attend to carried memory (if any).
                h = exp.encode_with_cross_attn(ids, carried)  # [B, T, d]
            else:
                # Seed-prepend path.
                h = exp.encode_with_seed(ids, carried)  # [B, (K or 0) + T, d]
            logits = exp.logits_from_hidden(h[:, -1, :])

            # Mask out switch-to-self (no-op) to avoid trivial loops.
            self_switch = tok.switch_id(active)
            logits[:, self_switch] = float("-inf")
            # Enforce the switch budget.
            if switch_count >= max_switches:
                for _n in self.expert_names:
                    logits[:, tok.switch_id(_n)] = float("-inf")

            # Apply temperature + top-k.
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # [B,1]
            nxt_id = int(nxt.item())

            # Check for switch / eos.
            if nxt_id == tok.eos_id:
                break
            if nxt_id in {tok.switch_id(n) for n in self.expert_names}:
                # Determine target expert.
                target = None
                for n in self.expert_names:
                    if nxt_id == tok.switch_id(n):
                        target = n
                        break
                if target is None or target == active:
                    # Shouldn't happen, but guard anyway.
                    continue
                # Carry the last k hidden states through the shared space.
                # Reuse the h we already computed this step (no re-encode).
                k_next = min(K, h.size(1))
                carried = self._carry_through_shared(
                    exp, h[:, -k_next:, :], self.expert(target),
                )
                switch_count += 1
                active = target
                # Start the new expert from a minimal context: the carried
                # states seed its first positions. We do NOT re-encode the old
                # history in the new tokenizer (that was lossy and O(T) per
                # switch); the carried states are the hand-off signal.
                ids = torch.empty((1, 0), dtype=torch.long, device=device)
                continue

            out_tokens.append((active, nxt_id))
            ids = torch.cat([ids, nxt], dim=1)

        # Decode per-expert segments for a faithful rendering.
        return self._render(out_tokens)

    def _render(self, out_tokens: List[Tuple[str, int]]) -> str:
        """Decode token runs, grouping consecutive tokens by expert."""
        if not out_tokens:
            return ""
        segments: List[str] = []
        cur_expert = out_tokens[0][0]
        cur_ids: List[int] = []
        for expert_name, tid in out_tokens:
            if expert_name != cur_expert:
                segments.append(
                    f"[{cur_expert}] {self.tokenizers[cur_expert].decode(cur_ids)}"
                )
                cur_expert = expert_name
                cur_ids = []
            cur_ids.append(tid)
        segments.append(f"[{cur_expert}] {self.tokenizers[cur_expert].decode(cur_ids)}")
        return "\n".join(segments)
