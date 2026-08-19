# Scenario Family Taxonomy

Initial mechanism families should include:

```text
character insertion, deletion, substitution, and transposition
Unicode variation and transliteration
token reordering and punctuation changes
date shifts and month/day ambiguity
source-specific and informative missingness
frequency skew and common-value collisions
field dependence and irrelevant variables
shared household-like attributes
within-source duplication
source-only no-match records
candidate ambiguity and near ties
one-to-one, many-to-one, and one-to-many conflicts
source-size imbalance
label scarcity and label noise
three-or-more-source graph contradictions
cannot-link conflicts
```

Each family receives a stable ID, mechanism tags, latent manifest digest, observable profile
schema version, and prospective holdout status. Composite realistic families may combine
several mechanisms but must retain their component tags.
