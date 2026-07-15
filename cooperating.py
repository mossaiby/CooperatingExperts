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

    def _encode_segment(
        self,
        expert: Expert,
        ids: torch.Tensor,
        carried: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode a segment, choosing the CALM cross-attn path or the
        seed-prepend path based on config.

        `carried` is the receiving expert's own-d_model representation of the
        sender's last K hidden states (already produced by
        _carry_through_shared). It is [B, K, d_model] or None.

        Returns the last-layer hidden states of the receiving expert over the
        real tokens of `ids` -- shape [B, T, d_model] (no seed prefix), so
        the caller can index logits with a uniform h[:, -1, :] / h[:, -k:, :]
        regardless of which path was taken.
        """
        if carried is None:
            return expert.encode_with_cross_attn(ids, None) \
                if self._cross_attn_enabled() else expert.encode(ids)

        if self._cross_attn_enabled():
            # CALM path: cross-attend to the carried memory. No seed prefix,
            # so the output is [B, T, d] directly.
            return expert.encode_with_cross_attn(ids, carried)
        # Legacy path: prepend carried states as virtual seed positions.
        # Output is [B, K + T, d]; callers offset by k-1 to get the T logits.
        return expert.encode_with_seed(ids, carried)

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
        z = exp_a.to_shared_space(seed_src)       # [B, k, shared_dim]
        seed_b = exp_b.from_shared_space(z)       # [B, k, d_b]

        # 4. Run B on ids_b. Two paths:
        #    - CALM cross-attn: B's own hidden states cross-attend to seed_b
        #      as memory; output is [B, Tb, d_b] (no seed prefix), so the
        #      logits that predict ids_b are h_b[:, :Tb, :] shifted by one.
        #    - Legacy seed-prepend: seed_b is prepended as virtual positions
        #      0..k-1; output is [B, k+Tb, d_b] and the Tb predicting logits
        #      are h_b[:, k-1:k-1+Tb, :].
        if self._cross_attn_enabled():
            h_b = exp_b.encode_with_cross_attn(ids_b, seed_b)  # [B, Tb, d_b]
            Tb = ids_b.size(1)
            logits_b = exp_b.logits_from_hidden(h_b[:, :-1, :])  # [B, Tb-1, V_b]
            targets_b = ids_b[:, 1:]                              # [B, Tb-1]
            # For the alignment regularizer we need B's own hidden over its
            # real tokens (without the cross-attn residual). Re-derive from
            # the pre-cross-attn representation would require a second pass;
            # instead use h_b (post cross-attn) detached -- the round-trip
            # regularizer only constrains the projection matrices, and using
            # the refined states is a valid (if slightly stronger) target.
            ref_b_hidden = h_b.detach()
        else:
            h_b = exp_b.encode_with_seed(ids_b, seed_b)  # [B, k + Tb, d_b]
            Tb = ids_b.size(1)
            logits_b = exp_b.logits_from_hidden(h_b[:, k - 1:k - 1 + Tb, :])
            targets_b = ids_b
            ref_b_hidden = h_b[:, k:, :].detach()       # [B, Tb, d_b] = B's own hidden

        pad_b = self.tokenizers[name_b].pad_id
        lm_loss = F.cross_entropy(
            logits_b.reshape(-1, logits_b.size(-1)),
            targets_b.reshape(-1),
            ignore_index=pad_b,
        )

        # 6. Alignment regularizer: encourage each expert's own round-trip
        #    (from_shared . to_shared) to be close to identity. We reuse hidden
        #    states already computed (A's boundary states, and B's own hidden
        #    over its real tokens taken from h_b) instead of running a second
        #    forward pass through B. The references are detached so this term
        #    only trains the projection weights, not the (here frozen) blocks.
        ref_a = seed_src.detach()
        align_a = F.mse_loss(
            exp_a.from_shared_space(exp_a.to_shared_space(ref_a)), ref_a
        )
        align_b = F.mse_loss(
            exp_b.from_shared_space(exp_b.to_shared_space(ref_b_hidden)), ref_b_hidden
        )
        align = align_a + align_b

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

            # Seed the segment with the carried states from the previous
            # expert. For the FIRST segment we seed with K zero vectors so
            # every segment uses the same "index k-1 predicts ids[0]" offset
            # (legacy path) or, with cross-attn, a zero memory bank.
            if carried is None:
                carried = torch.zeros(B, K, exp.cfg.d_model, device=device)
            k = carried.size(1)

            if self._cross_attn_enabled():
                # CALM path: cross-attend to carried memory; output is
                # [B, T, d] with no seed prefix, so logits predicting ids
                # are h[:, :-1, :] vs targets ids[:, 1:].
                h = exp.encode_with_cross_attn(ids, carried)  # [B, T, d]
                logits = exp.logits_from_hidden(h[:, :-1, :])  # [B, T-1, V]
                targets = ids[:, 1:]                            # [B, T-1]
            else:
                # Legacy seed-prepend path.
                h = exp.encode_with_seed(ids, carried)  # [B, k + T, d]
                logits = exp.logits_from_hidden(h[:, k - 1:k - 1 + T, :])  # [B, T, V]
                targets = ids                              # [B, T]

            pad = self.tokenizers[name].pad_id

            # Down-weight switch tokens to prevent over-switching.
            sw_weight = torch.ones(exp.vocab_size, device=device)
            sw_w = self.config.train.switch_loss_weight
            for _n in self.expert_names:
                sw_weight[self.tokenizers[name].switch_id(_n)] = sw_w

            seg_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad,
                weight=sw_weight,
            )
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

            # Encode the current sequence ONCE, seeded by the carried states
            # (if any). We keep the full last-layer hidden `h` so that, on a
            # switch, we can reuse its tail instead of re-encoding.
            if self._cross_attn_enabled():
                # CALM path: cross-attend to carried memory (if any). When
                # ids is empty (just switched, no tokens yet) we cannot
                # cross-attend -- fall back to a single pad token so the
                # expert produces a query position to predict from.
                if T == 0:
                    ids = torch.tensor([[tok.pad_id]], dtype=torch.long,
                                       device=device)
                    T = 1
                h = exp.encode_with_cross_attn(ids, carried)  # [B, T, d]
            else:
                # Legacy seed-prepend path.
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
