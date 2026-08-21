"""
model.py -- COMPOSE architecture, adapted to run on discrete medical
codes instead of free text.

Everything in this file is a direct, documented adaptation of the
original COMPOSE model (Gao et al., KDD 2020, model.py):

  CausalConv1d, HighwayBlock, ECEmbedding : UNCHANGED from the original.
      ECEmbedding is COMPOSE's text encoder: 4 parallel causal
      convolutions (kernel sizes 1/3/5/7) -> 3 highway blocks -> masked
      max-pool. In the original it consumes BERT word embeddings for a
      trial-criterion sentence. Here we reuse it AS-IS to encode a short
      TOKEN SEQUENCE per criterion (entity_type, code, operator, value),
      built from trained nn.Embedding lookups instead of BERT.

  CodeSetPooling : NEW. Original COMPOSE precomputes, for each of its 12
      fixed (diagnosis/procedure/product x 4-hierarchy-level) slots, a
      single BERT-derived word_dim vector per visit. Our processed data
      has no 4-level hierarchy text, only bare codes with a *variable*
      number of codes per category per visit, so we replace that
      precomputation with a masked mean-pool over learned code
      embeddings for each of our 3 categories (diagnosis, medication,
      lab). This produces the same shape the memory network expects:
      one word_dim vector per category per visit.

  EHRMemoryNetwork : adapted from the original 12-slot memory bank to a
      3-slot one (diagnosis, medication, lab instead of 12 hierarchy
      slots), matching our 3 code categories. The erase/add recurrence
      over visits is otherwise identical to the original.

  QueryNetwork, get_loss : UNCHANGED logic from the original -- same
      attention-based read from memory, same 3-way classification head,
      same masked CosineEmbeddingLoss on (response, query) using the
      match/mismatch/unknown label convention.
"""
import torch
from torch import nn
import torch.nn.functional as F

from config import config


# ----------------------------------------------------------------------
# UNCHANGED from original COMPOSE (Gao et al., 2020, model.py)
# ----------------------------------------------------------------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                          padding=0, dilation=dilation, groups=groups, bias=bias)
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super().forward(F.pad(input, (self.__padding, 0)))


class HighwayBlock(nn.Module):
    def __init__(self, input_dim, kernel_size):
        super().__init__()
        self.conv_t = CausalConv1d(input_dim, input_dim, kernel_size)
        self.conv_z = CausalConv1d(input_dim, input_dim, kernel_size)

    def forward(self, input):
        t = torch.sigmoid(self.conv_t(input))
        z = t * self.conv_z(input) + (1 - t) * input
        return z


class ECEmbedding(nn.Module):
    """Original COMPOSE text encoder. Consumes (B, L, word_dim) + (B, L) mask,
    returns a pooled (B, 4*conv_dim) representation. Used here to encode the
    4-token criterion sequence (see dataset.py)."""

    def __init__(self, word_dim, conv_dim):
        super().__init__()
        self.word_dim = word_dim
        self.conv_dim = conv_dim
        self.init_conv1 = CausalConv1d(word_dim, conv_dim, 1)
        self.init_conv2 = CausalConv1d(word_dim, conv_dim, 3)
        self.init_conv3 = CausalConv1d(word_dim, conv_dim, 5)
        self.init_conv4 = CausalConv1d(word_dim, conv_dim, 7)
        self.highway1 = HighwayBlock(4 * conv_dim, 3)
        self.highway2 = HighwayBlock(4 * conv_dim, 3)
        self.highway3 = HighwayBlock(4 * conv_dim, 3)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input, mask):
        input = input.permute(0, 2, 1)
        conv1 = self.init_conv1(input)
        conv2 = self.init_conv2(input)
        conv3 = self.init_conv3(input)
        conv4 = self.init_conv4(input)
        concat = torch.cat((conv1, conv2, conv3, conv4), dim=1)

        h = torch.relu(self.highway1(concat))
        h = torch.relu(self.highway2(h))
        h = torch.relu(self.highway3(h))

        h = h * mask.unsqueeze(1)
        pooled = self.pool(h).squeeze(-1)
        return pooled


# ----------------------------------------------------------------------
# NEW: replaces original COMPOSE's precomputed BERT code embeddings
# ----------------------------------------------------------------------
class CodeSetPooling(nn.Module):
    """Masked mean-pool of learned code embeddings -> one word_dim vector
    per (category, visit). One embedding table per category so diagnosis
    code 1234 and medication code 1234 do not collide."""

    def __init__(self, vocab_sizes: dict, word_dim: int, category_order):
        super().__init__()
        self.category_order = category_order
        self.embeddings = nn.ModuleDict({
            cat: nn.Embedding(vocab_sizes[cat], word_dim, padding_idx=0) for cat in category_order
        })

    def forward(self, ehr_ids, ehr_mask):
        # ehr_ids/ehr_mask: (B, T, num_categories, K)
        pooled_per_cat = []
        for cat_idx, cat in enumerate(self.category_order):
            ids = ehr_ids[:, :, cat_idx, :]     # (B, T, K)
            mask = ehr_mask[:, :, cat_idx, :]   # (B, T, K)
            emb = self.embeddings[cat](ids)      # (B, T, K, word_dim)
            summed = (emb * mask.unsqueeze(-1)).sum(dim=2)
            denom = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            pooled = summed / denom              # (B, T, word_dim)
            pooled_per_cat.append(pooled)
        return torch.stack(pooled_per_cat, dim=2)  # (B, T, num_categories, word_dim)


# ----------------------------------------------------------------------
# Adapted from original COMPOSE EHRMemoryNetwork (12 slots -> num_categories slots)
# ----------------------------------------------------------------------
class EHRMemoryNetwork(nn.Module):
    def __init__(self, word_dim, mem_dim, demo_dim, num_slots):
        super().__init__()
        self.mem_dim = mem_dim
        self.num_slots = num_slots

        self.erase_layer = nn.Linear(word_dim, mem_dim)
        self.add_layer = nn.Linear(word_dim, mem_dim)
        self.demo_embd = nn.Linear(demo_dim, mem_dim)

        self.init_memory = nn.Parameter(torch.randn(num_slots, mem_dim))

    def forward(self, input, demo, visit_mask):
        # input: (B, T, num_slots, word_dim); visit_mask: (B, T)
        batch_size, time_step, num_slots, word_dim = input.shape
        assert num_slots == self.num_slots

        memory = self.init_memory.unsqueeze(0).repeat(batch_size, 1, 1)
        demo_mem = torch.tanh(self.demo_embd(demo))

        for t in range(time_step):
            cur_input = input[:, t, :, :].reshape(batch_size * num_slots, word_dim)
            erase = torch.sigmoid(self.erase_layer(cur_input)).reshape(batch_size, num_slots, self.mem_dim)
            add = torch.tanh(self.add_layer(cur_input)).reshape(batch_size, num_slots, self.mem_dim)
            cur_mask = visit_mask[:, t].reshape(batch_size, 1, 1)
            erase = erase * cur_mask
            add = add * cur_mask
            memory = memory * (1 - erase) + add

        memory = torch.cat((memory, demo_mem.unsqueeze(1)), dim=1)  # (B, num_slots+1, mem_dim)
        return memory


# ----------------------------------------------------------------------
# UNCHANGED logic from original COMPOSE QueryNetwork
# ----------------------------------------------------------------------
class QueryNetwork(nn.Module):
    def __init__(self, mem_dim, conv_dim, mlp_dim):
        super().__init__()
        self.word_trans = nn.Linear(4 * conv_dim, mem_dim, bias=False)
        self.mlp = nn.Linear(2 * mem_dim, mlp_dim)
        self.output = nn.Linear(mlp_dim, 3)

    def forward(self, memory, query):
        trans_query = torch.relu(self.word_trans(query))          # (B, mem_dim)
        attention = torch.bmm(trans_query.unsqueeze(1), memory.permute(0, 2, 1)).squeeze(1)  # (B, num_slots+1)
        attention = torch.softmax(attention, dim=-1)
        response = torch.mean(attention.unsqueeze(-1) * memory, dim=1, keepdim=False)  # (B, mem_dim)

        out = torch.cat((response, trans_query), dim=-1)
        out = torch.relu(self.mlp(out))
        out = self.output(out)
        return out, response, trans_query, attention


# ----------------------------------------------------------------------
# Full model wrapper
# ----------------------------------------------------------------------
class ComposeModel(nn.Module):
    def __init__(self, vocabs: dict):
        super().__init__()
        category_order = ["diagnosis", "medication", "lab"]
        vocab_sizes = {cat: len(vocabs[cat]) for cat in category_order}
        token_vocab_size = max(len(vocabs["entity_type"]), len(vocabs["operator"]),
                                len(vocabs["diagnosis"]), len(vocabs["medication"]), len(vocabs["lab"]))

        self.code_pool = CodeSetPooling(vocab_sizes, config.CODE_EMBED_DIM, category_order)
        self.ehr_network = EHRMemoryNetwork(config.CODE_EMBED_DIM, config.MEM_DIM,
                                             config.DEMO_DIM, num_slots=len(category_order))

        # Separate embedding tables for criterion tokens (mirrors CodeSetPooling's
        # per-category tables, plus dedicated tables for entity_type/operator).
        self.entity_type_embd = nn.Embedding(len(vocabs["entity_type"]), config.CODE_EMBED_DIM, padding_idx=0)
        self.operator_embd = nn.Embedding(len(vocabs["operator"]), config.CODE_EMBED_DIM, padding_idx=0)
        self.diag_embd = nn.Embedding(len(vocabs["diagnosis"]), config.CODE_EMBED_DIM, padding_idx=0)
        self.med_embd = nn.Embedding(len(vocabs["medication"]), config.CODE_EMBED_DIM, padding_idx=0)
        self.lab_embd = nn.Embedding(len(vocabs["lab"]), config.CODE_EMBED_DIM, padding_idx=0)
        self.value_proj = nn.Linear(1, config.CODE_EMBED_DIM)

        self.ec_network = ECEmbedding(config.CODE_EMBED_DIM, config.CONV_DIM)
        self.query_network = QueryNetwork(config.MEM_DIM, config.CONV_DIM, config.MLP_DIM)

    def encode_ehr(self, ehr_ids, ehr_mask, visit_mask, demo):
        pooled = self.code_pool(ehr_ids, ehr_mask)          # (B, T, 3, word_dim)
        memory = self.ehr_network(pooled, demo, visit_mask)  # (B, 4, mem_dim)
        return memory

    def encode_criterion(self, crit_ids, crit_mask, crit_value, entity_type_of_batch):
        """crit_ids: (B, 4) -> [entity_type_id, code_id, operator_id, value_slot_id]
        entity_type_of_batch: list[str] length B, used to route the code id
        (index 1) through the correct per-category embedding table."""
        B = crit_ids.size(0)
        et_tok = self.entity_type_embd(crit_ids[:, 0])   # (B, word_dim)

        code_tok = torch.zeros(B, config.CODE_EMBED_DIM, device=crit_ids.device)
        for i in range(B):
            et = entity_type_of_batch[i]
            table = {"diagnosis": self.diag_embd, "medication": self.med_embd, "lab": self.lab_embd}.get(
                et, self.diag_embd)
            code_tok[i] = table(crit_ids[i, 1])

        op_tok = self.operator_embd(crit_ids[:, 2])       # (B, word_dim)
        val_tok = self.operator_embd(crit_ids[:, 3]) + self.value_proj(crit_value)  # (B, word_dim)

        tokens = torch.stack([et_tok, code_tok, op_tok, val_tok], dim=1)  # (B, 4, word_dim)
        criteria_embd = self.ec_network(tokens, crit_mask)                # (B, 4*conv_dim)
        return criteria_embd

    def forward(self, batch, entity_type_of_batch):
        memory = self.encode_ehr(batch["ehr_ids"], batch["ehr_mask"], batch["visit_mask"], batch["demo"])
        criteria_embd = self.encode_criterion(batch["crit_ids"], batch["crit_mask"], batch["crit_value"],
                                               entity_type_of_batch)
        output, response, query, attention = self.query_network(memory, criteria_embd)
        return output, response, query, attention


def get_loss(output, response, query, label, device):
    """UNCHANGED logic from original COMPOSE get_loss."""
    similarity_label, label_mask = [], []
    for lb in label.tolist():
        if lb == 0:
            similarity_label.append(1); label_mask.append(1)
        elif lb == 1:
            similarity_label.append(-1); label_mask.append(1)
        else:
            similarity_label.append(1); label_mask.append(0)

    similarity_label = torch.tensor(similarity_label, dtype=torch.long, device=device)
    label_mask = torch.tensor(label_mask, dtype=torch.float32, device=device)

    ce_loss = nn.CrossEntropyLoss()
    sm_loss = nn.CosineEmbeddingLoss(margin=0.3, reduction="none")

    pred = torch.softmax(output, dim=-1)
    loss = ce_loss(output, label)

    similarity = sm_loss(response, query, similarity_label)
    similarity = similarity * label_mask
    denom = torch.sum(label_mask).clamp(min=1.0)
    similarity = torch.sum(similarity) / denom

    total = loss + config.SIMILARITY_LOSS_WEIGHT * similarity
    return total, loss, similarity, pred
