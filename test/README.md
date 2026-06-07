# Lightweight Tests

The ICDM reviewer branch keeps only tests that run without external datasets or
the full GPU baseline stack.

```bash
make test
make unit
make verify
```

- `make test` runs the dev-container smoke import check.
- `make unit` runs focused unit tests for the sparse helper, optimizer,
  launcher, and DF-Leiden wrapper behavior.
- `make verify` also regenerates the two article figures from
  `results/icdm-2026-1`.
