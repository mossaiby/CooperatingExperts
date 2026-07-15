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

        # 1. Encode prefix with A, take last-position hidden state.
        h_full = exp_a.encode(ids_a)          # [B, Ta, d_a]
        h_a_last = h_full[:, -1, :]           # [B, d_a]

        # 2-3. Carry through shared space into B's hidden space.
        z = exp_a.to_shared_space(h_a_last)   # [B, shared_dim]
        h_b0 = exp_b.from_shared_space(z)      # [B, d_b]

        # 4. Run B on ids_b, prepending the carried state as a virtual token.
        #    We do this by embedding ids_b, then concatenating h_b0 as the
        #    first position and running the transformer blocks over the
        #    combined sequence.
        x_b = exp_b._embed(ids_b)             # [B, Tb, d_b]
        x_b = torch.cat([h_b0.unsqueeze(1), x_b], dim=1)  # [B, Tb+1, d_b]
        h_b = exp_b._blocks(x_b)             # [B, Tb+1, d_b]

        # 5. LM loss on B's tokens. The prediction at position i (of the
        #    Tb+1 sequence) predicts ids_b[i]. So we use h_b[:, :-1, :] to
        #    predict ids_b (shifted by one because of the prepended state).
        logits_b = exp_b.logits_from_hidden(h_b[:, :-1, :])  # [B, Tb, V_b]
        targets_b = ids_b  # [B, Tb]
        pad_b = self.tokenizers[name_b].pad_id
        lm_loss = F.cross_entropy(
            logits_b.reshape(-1, logits_b.size(-1)),
            targets_b.reshape(-1),
            ignore_index=pad_b,
        )

        # 6. Alignment regularizer (round-trip through A's own projections).
        align_loss = F.mse_loss(
            exp_a.from_shared_space(exp_a.to_shared_space(h_a_last)),
            h_a_last,
        )
        # Same for B — FIX: no no_grad() so B's projection weights
        # actually receive gradients from align_loss_b.
        h_b_ref = exp_b.encode(ids_b)[:, -1, :]
        align_loss_b = F.mse_loss(
            exp_b.from_shared_space(exp_b.to_shared_space(h_b_ref)),
            h_b_ref,
        )
        align = align_loss + align_loss_b

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
        carried_h: Optional[torch.Tensor] = None  # [B, d] carried hidden

        for seg_idx, (name, ids) in enumerate(segments):
            exp = self.expert(name)
            ids = ids.to(device)
            # Ensure a batch dimension: [T] -> [1, T].
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            B, T = ids.shape
            if T > exp.cfg.max_seq_len:
                ids = ids[:, -exp.cfg.max_seq_len:]

            # Embed; if we carry a hidden state from the previous expert,
            # prepend it as a virtual first position (same as generation).
            # FIX: unified carried-state path — zeros for first segment
            # so ALL segments use the same T-token prediction offset.
            if carried_h is None:
                carried_h = torch.zeros(B, exp.cfg.d_model, device=device)

            x = exp._embed(ids)
            x = torch.cat([carried_h.unsqueeze(1), x], dim=1)  # [B, T+1, d]
            h = exp._blocks(x)
            logits  = exp.logits_from_hidden(h[:, :T, :])  # [B, T, V]
            targets = ids                                    # [B, T]

            pad = self.tokenizers[name].pad_id

            # FIX: down-weight switch tokens to prevent over-switching.
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

            # Carry the LAST hidden state through the shared space to the next
            # expert (if any). Use the last real position (ignore the prepended
            # virtual one by taking index -1 of h). We DETACH the carried
            # state so gradients do not flow back through the previous expert's
            # entire transformer via from_shared — this (a) bounds memory for
            # long batch=1 sessions, and (b) matches generation, where the
            # carried state is produced under torch.no_grad().
            h_last = h[:, -1, :]  # [B, d]
            z = exp.to_shared_space(h_last)
            # The next segment's expert is unknown here; we store z and let the
            # next iteration project it back. To do that we need the next
            # expert's from_shared. We peek at the next segment's name.
            next_name = segments[seg_idx + 1][0] if seg_idx + 1 < len(segments) else None
            if next_name is not None:
                # Detach so the next segment's forward does not backprop
                # through this segment's graph (see comment above).
                carried_h = self.expert(next_name).from_shared_space(z).detach()
            else:
                carried_h = None

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

        # Carried hidden state (None until a switch happens).
        carried_h: Optional[torch.Tensor] = None
        switch_count: int = 0

        out_tokens: List[Tuple[str, int]] = []  # (expert, token_id)
        eos_set = {tok.eos_id for tok in self.tokenizers.values()}

        for _ in range(max_new_tokens):
            exp = self.expert(active)
            tok = self.tokenizers[active]
            T = ids.size(1)
            if T > exp.cfg.max_seq_len:
                ids = ids[:, -exp.cfg.max_seq_len:]

            # Encode current sequence ONCE, optionally prepending a carried
            # state. We keep the full last-layer hidden `h` so that, on a
            # switch, we can reuse h[:, -1, :] instead of re-encoding.
            x = exp._embed(ids)
            if carried_h is not None:
                x = torch.cat([carried_h.unsqueeze(1), x], dim=1)
            h = exp._blocks(x)  # [B, T(+1), d]
            logits = exp.logits_from_hidden(h[:, -1, :])

            # Mask out switch-to-self (no-op) to avoid trivial loops.
            self_switch = tok.switch_id(active)
            logits[:, self_switch] = float("-inf")
            # FIX: enforce switch budget.
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
                # Carry the last hidden state through the shared space.
                # Reuse the h we already computed this step (no re-encode).
                h_last = h[:, -1, :]  # [B, d]
                z = exp.to_shared_space(h_last)
                carried_h = self.expert(target).from_shared_space(z)
                switch_count += 1
                active = target
                # Start the new expert from a minimal context: just the
                # carried state seeds its first hidden position. We do NOT
                # re-encode the old history in the new tokenizer (that was
                # lossy and O(T) per switch); the carried state is the
                # hand-off signal, by design.
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
