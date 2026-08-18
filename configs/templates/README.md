# Local configuration templates

These files are **worksheets**, not committed operational configurations.

Copy the relevant template into `private/config/`, complete it only in the authorised
local environment, validate it with `mapel-linkage validate-config`, and keep the
completed file Git-ignored.

Never place real source names, identifiers, secrets, adjudication values, crosswalk
content, or operational paths in a Git-tracked template. An unverified crosswalk is
reference evidence only and is ineligible for training, calibration, threshold
selection, or testing.
