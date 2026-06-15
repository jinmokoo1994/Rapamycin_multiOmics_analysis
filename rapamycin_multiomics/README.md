# Rapamycin Multi-Omics: Evidence for Therapeutic Effect

A reproducible Jupyter-notebook series that assembles **real, published, open-access
multi-omics data** on rapamycin (sirolimus) — an mTOR inhibitor — and analyzes it to
characterize the molecular basis of its therapeutic effects (longevity / mTOR-pathway
inhibition / anti-cancer drug sensitivity).

All datasets are downloaded programmatically from public repositories. No proprietary data.

## Omics layers & data sources

| Layer | Dataset | Source | What it shows |
|-------|---------|--------|---------------|
| **Target / mechanism (genomics)** | Rapamycin → FKBP1A / MTOR bioactivity | ChEMBL (EMBL-EBI) | Molecular target & potency of rapamycin |
| **Transcriptomics** | GSE131754 — mouse liver RNA-seq, rapamycin vs control (12 vs 12) | NCBI GEO; Tyshkovskiy et al., *Cell Metabolism* 2019, [10.1016/j.cmet.2019.06.018](https://doi.org/10.1016/j.cmet.2019.06.018) | Rapamycin gene-expression / longevity signature |
| **Pharmacogenomics** | GDSC1 — Temsirolimus (rapamycin prodrug, CCI-779) IC50 across ~970 cancer cell lines | Therapeutics Data Commons (`tdc.multi_pred.DrugRes`) | Anti-cancer drug response & sensitive lineages |
| **Proteomics / phosphoproteomics** | PXD067812 — rapamycin-treated human DLBCL (SU-DHL-4), TMT-18plex global + phospho | PRIDE/ProteomeXchange | Protein & phosphosite-level mTOR pathway response |

## Notebooks

- `00_data_acquisition.ipynb` — download & cache every dataset, document provenance.
- `01_target_mechanism.ipynb` — rapamycin's molecular target (FKBP1A/MTOR), mechanism, the mTOR pathway.
- `02_transcriptomics.ipynb` — differential expression (rapamycin vs control), mTOR/longevity pathway enrichment.
- `03_pharmacogenomics.ipynb` — GDSC Temsirolimus sensitivity across cancer lineages, biomarkers of response.
- `04_proteomics.ipynb` — global + phospho proteome response to rapamycin, mTOR-substrate signaling.
- `05_integration.ipynb` — cross-omics synthesis: a coherent therapeutic-evidence narrative.

## Reproduce

```bash
cd rapamycin_multiomics
pip install -r requirements.txt
jupyter lab          # run notebooks 00 -> 05 in order
```

Notebook `00` populates `data/raw/` and `data/processed/`; later notebooks read from there.

## Scientific framing & honesty

These analyses present **molecular evidence consistent with rapamycin's known therapeutic
mechanisms** (mTOR inhibition, transcriptional longevity signature, cancer-cell growth
inhibition). They are secondary analyses of published data and are **not** a substitute for
controlled clinical efficacy trials. Claims are scoped to what each dataset can support.

## Citations

According to PubMed and the source repositories:
- Tyshkovskiy A, et al. *Cell Metab* 2019;30(3):573-593. https://doi.org/10.1016/j.cmet.2019.06.018 (GEO GSE131754)
- GDSC drug response via Therapeutics Data Commons (Huang et al., *Nat Chem Biol* 2022). https://tdcommons.ai
- PRIDE/ProteomeXchange PXD067812 (rapamycin-treated SU-DHL-4 quantitative proteomics).
- ChEMBL v34 (Zdrazil et al., *Nucleic Acids Res* 2024).
