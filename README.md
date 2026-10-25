# HINGE — Histology and Gene-guided Neighboring Cell-matching Ensemble

> Robust cross-panel cell matching for adjacent spatial transcriptomics sections
> using histomorphology, gene expression, and spatial proximity.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

## Overview

Spatial transcriptomics platforms measure gene expression in tissue sections, but a single panel is often insufficient to capture all genes of interest. A common strategy is to measure **adjacent tissue sections** with different gene panels and then integrate them by matching individual cells. However, even small misalignments between sections (~22 µm median) cause distance-only methods to fail — multiple morphologically similar cells lie within the same spatial radius.
<img width="2070" height="1466" alt="Fig1B_Distance_only_Matching_Fails_Even_Without_Cell_Dropout" src="https://github.com/user-attachments/assets/0c39d9c4-3929-4f42-9543-4ba4d06ffac4" />

**HINGE** solves this by combining three complementary signals:

| Signal | Source |
|---|---|
| Histomorphology | UNI pathology foundation model |
| Gene expression | Overlapping genes, quality-filtered |
| Spatial proximity | Candidate filtering by NN distance |

---
<img width="1492" height="840" alt="Screenshot 2026-10-25 191031" src="https://github.com/user-attachments/assets/233b6679-3ede-4f7e-9023-f3e352f3fd93" />
## Hinge improves distance and gene expression based matching
<img width="2370" height="1466" alt="Fig4_hinge_main_result_bar_sd" src="https://github.com/user-attachments/assets/75a86dc9-07df-4975-8e2a-e9e0f1207c25" />
