"""Continuous training is disabled until feedback has independent human review.

This intentionally does not read/write the production DB or reference CSV.
Use a separately prepared, reviewed article dataset and ml.train to generate
a candidate. Votes, including -1 disagreements, must never become labels.
"""
import logging

logger = logging.getLogger(__name__)


def execute_continuous_training():
    raise RuntimeError(
        "Retreino automático com votos não revisados foi desativado. "
        "Prepare um corpus revisado com título, corpo, datas e evidências; "
        "execute python -m ml.train --data <corpus.csv>. Apenas candidatos, sem promoção."
    )


if __name__ == "__main__":
    execute_continuous_training()
