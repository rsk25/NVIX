from typing import Tuple, Any

import torch
from torch import nn

import random

from common.const.model import DEF_Q_HIDDEN, MDL_Q_HIDDEN, MDL_Q_HEAD, DEF_Q_EMBED
from common.const.operand import VOCAB_MAX
from common.const.pad import NEG_INF, PAD_ID
from common.data import Encoded, Label
from model.base.util import init_weights, logsoftmax
from model.ept.attention import MultiheadAttentionWeights


class PointerGeneratorHead(nn.Module):
    def __init__(self, hidden_dim: int = DEF_Q_HIDDEN, embed_dim: int = DEF_Q_EMBED, vocab_size: int = VOCAB_MAX,
                 init_factor: float = 0.01, debug: bool = False, num_head: int = 1):
        super().__init__()

        # Single-head attention score layer
        self.num_head = num_head
        self.encoder_attention = MultiheadAttentionWeights(**{MDL_Q_HIDDEN: hidden_dim, MDL_Q_HEAD: self.num_head})
        if hidden_dim != embed_dim:
            self.hidden_to_embed = torch.nn.Linear(hidden_dim, embed_dim)

        # Generation distribution layer
        self.generation_dist = torch.nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

        # W_h h^*_t + W_s s_t + W_x x_t + b
        self.generation_prob_linear = nn.Linear(in_features=hidden_dim * 2 + embed_dim, out_features=1)
        self.apply(lambda module: init_weights(module, init_factor))

        self.log_sigmoid = nn.LogSigmoid()

        #self.linear_v = nn.Linear(self.encoder_attention.hidden_dim, self.encoder_attention.hidden_dim)
        
        self.attn_reduce = nn.Linear(in_features=self.num_head, out_features=1)

        self.debug = debug
        self.intermediate_values = {}


    def _save_to_attribute(self, attr_key: str, value: torch.Tensor):
        value = value.cpu().detach()
        if attr_key not in self.intermediate_values:
            self.intermediate_values.update({attr_key: [value]})
        else:
            self.intermediate_values[attr_key].append(value)


    def _transform_value(self, value, batch_size):
            # Retrieve shape
            value_batch, key_len = value.shape[:2]

            # Pass linear and transpose value matrix: [1 or B, S, N, H/N] -> [1 or B, N, S, H/N].
            value = self.linear_v(value) \
                .view(value_batch, key_len, self.num_head, self.encoder_attention.dim_head).transpose(1, 2)

            # If value has shape [1, *], expand it.
            if value_batch == 1:
                value = value.expand(batch_size, -1, -1, -1)

            # [B, N, S, H/N]
            return value


    def _compute_attention(self, text: Encoded, decoded: Encoded,
                           prev_key: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        # Text: [B, S, H]
        # Decoded: [B, T, H]

        # Attention score: [B, T, S, N]
        attn_score, new_key = self.encoder_attention.forward(query=decoded.vector, key=text.vector,
                                                            key_ignorance_mask=text.pad, prev_key=prev_key,
                                                            head_at_last=True, is_self=False)

        if self.num_head != 1:
            attn_score = logsoftmax(attn_score)
            attn_score = self.attn_reduce(attn_score)
            
        # Apply softmax: [B, T, S, N=1] -> [B, T, S]
        attn_score = logsoftmax(attn_score.squeeze(-1))
        # Set score as zero on padding in decoded. (decoded.pad: [B, T])
        attn_score = attn_score.masked_fill(decoded.pad.unsqueeze(-1), NEG_INF)
        # Compute attented vector h_t^* [B, T, H] = [B, T, S] * [B, S, H]
        attented_vector = torch.bmm(attn_score.exp(), text.vector)
            
        # else:
        #     # print(f"attention shape (before logsoftmax): {attn_score.shape}")
        #     _batch_size, _key_len, _query_len, _ = attn_score.shape
        #     _embed_dim = text.vector.size(dim=2)

        #     # Apply softmax: [B, T, S, N]
        #     attn_score = logsoftmax(attn_score)
            
        #     # Set score as zero on padding in decoded. 
        #     attn_score = attn_score.masked_fill(decoded.pad.unsqueeze(-1).unsqueeze(-1), NEG_INF)
        #     # Reshape attention scores to [B, T, S, N] -> [B, N, T, S] -> [BN, T, S]
        #     attn_score_reshape = attn_score.permute(0, 3, 1, 2).flatten(0,1).contiguous()
        #     # Reshape text embedding to [B, S, H] -> [B, S, H/N, N] -> [B, N, S, H/N] -> [BN, S, H/N]
        #     new_value = self._transform_value(text.vector, _batch_size)
        #     value = new_value.flatten(0, 1).contiguous()

        #     # First, compute as [BN, T, S]*[BN, S, H/N] = [BN, T, H/N], 
        #     # then, reshape to [BN, T, H/N] -> [B, N, T, H/N] -> [B, T, H/N, N] -> [B, T, H]
        #     # Finally, reshape attented vector so that h_t^* [B, T, H] = [B, T, S] * [B, S, H]
        #     attented_vector = torch.bmm(attn_score_reshape.exp(), value) \
        #                            .view(_batch_size, self.num_head, _key_len, _embed_dim // self.num_head) \
        #                            .permute(0, 2, 3, 1).flatten(2, 3).contiguous()

        #     attn_score = self.attn_reduce(attn_score)
        #     attn_score = attn_score.squeeze(-1)

        return attented_vector, new_key, attn_score

    def _generation_probability(self, text_attn: torch.Tensor, decoded: torch.Tensor,
                                embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Text: [B, T, H]
        # Decoded: [B, T, H]
        # Embedding: [B, T, H]

        # [B, T, 3H]
        concatenated = torch.cat([text_attn, decoded, embedding], dim=-1)
        before_sigmoid = self.generation_prob_linear(concatenated)
        after_logsig = self.log_sigmoid(before_sigmoid)

        # Return transformed result [B, T, 1]
        return after_logsig, after_logsig - before_sigmoid

    def forward(self, text: Encoded, text_label: Label, decoded: Encoded, decoder_embedding: Encoded,
                prev_key: torch.Tensor = None, pad_value: int = 0) -> Tuple[torch.Tensor, tuple]:
        # Compute distribution of token generation
        if hasattr(self, 'hidden_to_embed'):
            decoded_vector = self.hidden_to_embed(decoded.vector)
        else:
            decoded_vector = decoded.vector
        gen_dist = logsoftmax(self.generation_dist(decoded_vector))

        # if text is None, then we cannot use copying method.
        if text is None:
            return gen_dist, (tuple(),)  # We need to return tuple to pass type checking.

        # Compute attention.
        # [B, T, H], ?, [B, T, S].
        text_attented, new_key, attn_score = self._compute_attention(text, decoded, prev_key=prev_key)

        # Compute generation probability
        gen_prob, copy_prob = self._generation_probability(text_attented, decoded.vector, decoder_embedding.vector)

        # Expand index to [B, T, S]
        text_label = text_label.indices

        # Copying probability
        #print(f"Before calculating copy_attn: attn_score={attn_score.shape}, copy_prob={copy_prob.shape}")
        copy_attn = (attn_score + copy_prob).exp()
        
        if torch.are_deterministic_algorithms_enabled():
            # Use manual but deterministic algorithm (scatter_add is non-deterministic on CUDA)
            copy_dist = torch.zeros_like(gen_dist)  # [B, T, V]
            batch_sz, text_len = text_label.shape
            for b in range(batch_sz):
                for s in range(text_len):
                    text_bs = text_label[b, s].item()
                    if text_bs == PAD_ID:
                        continue
                    copy_dist[b, :, text_bs] += copy_attn[b, :, s]
        else:
            copy_dist = torch.zeros_like(gen_dist).scatter_add(dim=-1, index=text_label.unsqueeze(1).expand(copy_attn.shape), src=copy_attn)

        # Generating probability (P_gen * Vocab)
        gen_dist = (gen_dist + gen_prob).exp()

        # Add copying to generation & return as log-probability
        logprob = (copy_dist + gen_dist).log()
        logprob = logprob.masked_fill(torch.isfinite(logprob).logical_not(), NEG_INF)

        if self.debug:
            logprob_centers = logprob[:, :, [101, 1996, 4578, 1997, 1996, 18493, 3578, 102]]
            for attr, val in zip(['attn_score', 'copy_prob', 'copy_attn', 'logprob'], [attn_score, copy_prob, copy_attn, logprob_centers]):
                self._save_to_attribute(attr, val)

        return logprob, (new_key,)
