"""Composition lattice over the six calibrated degradation families.

The coverage study needs a corruption sampler whose *support* is a set of pairs
rather than a set of severities. ``ShipDataset`` samples a severity within a
fixed kind; here the pair is the unit of sampling.

Only the six continuous-level families are used, at their measured matched-20pp
levels (``reports/composition/gate_target20.json``). The three severity-
parameterised families are deliberately excluded even though they carry the
paper's headline pair: at severity 5 they are not comparable in damage — haze
and cloud cost 1-2 pp of accuracy while glare costs 17.5 — so a lattice mixing
both regimes could not separate "more kinds of pair" from "harder pair". They
re-enter the study only as the *transfer* target.
"""
from __future__ import annotations

import itertools
from collections import Counter

from maritime_robust.corruptions_extra import FAMILIES, apply

# matched-20pp continuous levels, measured on the frozen fold0/s17 baseline
MATCHED_LEVEL = {"defocus": 0.41, "motion": 0.528, "noise": 0.435,
                 "rain": 0.392, "pixelate": 0.353, "impulse": 0.475}

# One pair per family pair, three held out: each family is excluded from exactly
# one held-out pair, so no family is under-trained relative to the others.
HELD_OUT = (("noise", "pixelate"), ("defocus", "rain"), ("motion", "impulse"))


def all_pairs(exclude=()):
    drop = {frozenset(p) for p in exclude}
    return [p for p in itertools.combinations(FAMILIES, 2) if frozenset(p) not in drop]


def balanced_order(pairs):
    """Order pairs greedily by family frequency.

    A nested pool must not be dominated by one family at short prefixes: the
    first 3 pairs of a random order can easily contain haze... or here, rain,
    three times, and then ``c3 -> c12`` would measure exposure to rain rather
    than the diversity of pairs.
    """
    count, left, order = Counter(), set(pairs), []
    while left:
        best = min(left, key=lambda p: (count[p[0]] + count[p[1]],
                                        max(count[p[0]], count[p[1]]),
                                        -abs(count[p[0]] - count[p[1]]), p))
        order.append(best)
        count[best[0]] += 1
        count[best[1]] += 1
        left.remove(best)
    return order


def apply_family(image, family, seed):
    return apply(image, family, MATCHED_LEVEL[family], seed)


def apply_pair(image, pair, seed):
    """Apply both families; the second seed is derived so the two draws are independent."""
    return apply_family(apply_family(image, pair[0], seed), pair[1], seed ^ 0x5BD1E995)


def pool(k):
    return balanced_order(all_pairs(HELD_OUT))[:k]


POOL_SIZE = len(all_pairs(HELD_OUT))
