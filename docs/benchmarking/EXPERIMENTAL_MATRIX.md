# Synthetic Benchmark Experimental Matrix

## Design axes

The core matrix varies, at minimum:

```text
corruption mechanism and intensity
missingness mechanism and intensity
frequency skew
field dependence
label volume and label noise
no-match prevalence
duplicate density
candidate-set ambiguity
source-size ratio
source count
linkage mode
assignment constraint
calibration sample size
review-capacity policy
runtime and memory budget
```

## Design components

1. fractional-factorial cells for estimable main effects and priority interactions;
2. space-filling coverage of continuous ranges;
3. stress families for difficult mechanisms;
4. composite realistic families;
5. prospectively held-out corruption mechanisms.

## Replicates

Seeds are replicates of an instance. Replicates estimate stochastic variability but do not
increase the independent scenario-family count.

## Corpus quality report

The registry reports family, instance, replicate, and run counts separately, along with
recipe-by-family coverage, pairwise comparison counts, failure rates, profile coverage, held-out
mechanisms, and learning-curve stability.
