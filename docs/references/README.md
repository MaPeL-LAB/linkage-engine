# References and Citation Style

The canonical bibliography is `references.bib`.

Markdown documents use citation keys in a Pandoc-compatible form:

```text
[@fellegi1969theory]
[@sadinle2017bayesian; @harron2017guide]
```

## Reference policy

- Prefer peer-reviewed papers, authoritative books, official documentation, and primary software repositories.
- Include DOI or stable official URL where available.
- Record access date for changeable software documentation.
- Do not cite transient ChatGPT/web-search reference identifiers in repository files.
- Add a new key only once and use a descriptive stable key.
- Software capability claims should be tied to versioned documentation or repository evidence.

The basic repository verification script checks duplicate keys and balanced braces. A full BibTeX parser should be added to CI when the dependency policy is finalized.
