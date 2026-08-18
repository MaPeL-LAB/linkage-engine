# Restricted output allow-list worksheet

Every exported field is deny-by-default.

| Output field | Relationship output | Review queue | Sensitivity classification | Purpose | Approved by |
|---|---:|---:|---|---|---|
| `relationship_id` | yes | yes | non-sensitive provenance | immutable relationship reference | `REPLACE_LOCALLY` |
| `REPLACE_LOCALLY` | yes/no | yes/no | `REPLACE_LOCALLY` | `REPLACE_LOCALLY` | `REPLACE_LOCALLY` |

Only fields approved in the local project configuration may be written to restricted
outputs. Unrestricted manifests remain aggregate-only.
