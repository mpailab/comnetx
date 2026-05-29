# cn69 Robustness Completion 2026-05-29

These scripts add only the missing paired `999:10` robustness measurements
needed to bring the selected rows to five stream-level runs:
Leiden, DF-Leiden, and S2CAG on `dyn_pubmed` and `arxivmath`.

Results are written to `results/paper_icdm/9` by default. Override
`PAPER_ICDM_SERIES` only if this series number is intentionally changed.

## Coverage

Existing paired counts in `results/registry/all_results.csv`:

- Leiden: `dyn_pubmed` 4/5, `arxivmath` 4/5.
- DF-Leiden: `dyn_pubmed` 3/5, `arxivmath` 4/5.
- S2CAG dataset features: `dyn_pubmed` 3/5, `arxivmath` 2/5.

New jobs:

1. `run01_arxivmath_s2cag_a.sh`: S2CAG `arxivmath`, repeat A.
2. `run02_arxivmath_s2cag_b.sh`: S2CAG `arxivmath`, repeat B.
3. `run03_arxivmath_s2cag_c.sh`: S2CAG `arxivmath`, repeat C.
4. `run04_pubmed_s2cag_and_arxiv_leiden.sh`: S2CAG `dyn_pubmed` repeat A plus Leiden `arxivmath`.
5. `run05_pubmed_s2cag_and_leiden.sh`: S2CAG `dyn_pubmed` repeat B plus Leiden `dyn_pubmed`.
6. `run06_arxivmath_dfleiden.sh`: DF-Leiden `arxivmath`.
7. `run07_pubmed_dfleiden_a.sh`: DF-Leiden `dyn_pubmed`, repeat A.
8. `run08_pubmed_dfleiden_b.sh`: DF-Leiden `dyn_pubmed`, repeat B.

The three `arxivmath` S2CAG jobs dominate runtime. Keeping them separate is the
most even schedule that avoids extra measurements beyond the target count of
five paired runs.

## Launch Lines

```bash
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run01_arxivmath_s2cag_a.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run02_arxivmath_s2cag_b.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run03_arxivmath_s2cag_c.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run04_pubmed_s2cag_and_arxiv_leiden.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run05_pubmed_s2cag_and_leiden.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run06_arxivmath_dfleiden.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run07_pubmed_dfleiden_a.sh'
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && PAPER_ICDM_SERIES=9 scripts/paper/cn69_robustness_20260529/run08_pubmed_dfleiden_b.sh'
```
