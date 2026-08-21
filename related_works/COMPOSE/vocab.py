"""
vocab.py -- builds integer vocabularies for every discrete thing the
COMPOSE-adaptation model needs to embed:

  * diagnosis codes   (ICD-10, from diagnoses_clean.parquet + trial criteria)
  * medication codes  (NDC,     from prescriptions_clean.parquet + trial criteria)
  * lab codes         (ITEMID,  from labs_clean.parquet + trial criteria)
  * entity types       ("diagnosis" | "medication" | "lab" | "procedure" | "administrative")
  * operators           (EXISTS, GT, LT, ... -- see trial_graph.Operator)

Codes seen only in trials but never in the patient population (and vice
versa) still get a vocabulary slot -- an unseen code simply gets a
near-random embedding, exactly as an unseen word would with any
embedding table. This is preferable to dropping the criterion, which
would silently change which trials are evaluable.
"""
import logging
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PAD = "<PAD>"
UNK = "<UNK>"


class Vocab:
    """Simple, picklable string<->index vocabulary with PAD=0, UNK=1."""

    def __init__(self):
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}

    def add(self, token: str):
        token = str(token)
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def add_many(self, tokens):
        for t in tokens:
            self.add(t)

    def __call__(self, token) -> int:
        return self.stoi.get(str(token), self.stoi[UNK])

    def __len__(self):
        return len(self.itos)


def build_vocabs(diag_df: pd.DataFrame, rx_df: pd.DataFrame, labs_df: pd.DataFrame, trial_store):
    """Build the four code-family vocabularies plus operator/entity-type vocabs.

    trial_store: a trial_graph.TrialStore (or compose_based-compatible store)
    """
    diag_vocab, med_vocab, lab_vocab = Vocab(), Vocab(), Vocab()
    entity_type_vocab, operator_vocab = Vocab(), Vocab()

    # --- from patient tables -----------------------------------------
    diag_col = "ICD10_CODE" if "ICD10_CODE" in diag_df.columns else "ICD9_CODE"
    diag_vocab.add_many(diag_df[diag_col].astype(str).unique())

    med_col = "NDC" if "NDC" in rx_df.columns else None
    if med_col:
        med_vocab.add_many(rx_df[med_col].astype(str).unique())

    lab_col = "ITEMID" if "ITEMID" in labs_df.columns else None
    if lab_col:
        lab_vocab.add_many(labs_df[lab_col].astype(str).unique())

    # --- from trial criteria -------------------------------------------
    for trial in trial_store.values():
        for c in list(trial.inclusion_criteria) + list(trial.exclusion_criteria):
            entity_type_vocab.add(c.entity_type)
            op_name = c.operator.value if hasattr(c.operator, "value") else str(c.operator)
            operator_vocab.add(op_name)
            if c.entity_type == "diagnosis":
                diag_vocab.add(c.entity_code)
            elif c.entity_type == "medication":
                med_vocab.add(c.entity_code)
            elif c.entity_type == "lab":
                lab_vocab.add(c.entity_code)
            # "procedure"/"administrative" criteria share no code vocab of
            # their own here -- they are embedded via entity_type + operator
            # tokens only, since our processed data has no procedure codes.

    logging.info(
        f"[Vocab] diagnosis={len(diag_vocab)} medication={len(med_vocab)} "
        f"lab={len(lab_vocab)} entity_type={len(entity_type_vocab)} operator={len(operator_vocab)}"
    )

    return {
        "diagnosis": diag_vocab,
        "medication": med_vocab,
        "lab": lab_vocab,
        "entity_type": entity_type_vocab,
        "operator": operator_vocab,
    }


def code_vocab_for(vocabs: dict, entity_type: str) -> Vocab:
    return vocabs.get(entity_type, vocabs["diagnosis"])
