# Configuration Schemas

`linkage-config.schema.json` is generated from `mapel_linkage.configuration.models.LinkageConfig` and is the normative machine-readable M1 configuration contract.

Regenerate it after any configuration-model change:

```bash
python scripts/generate_config_schema.py
```

The test suite compares the committed file with the live Pydantic schema and fails when they diverge.
