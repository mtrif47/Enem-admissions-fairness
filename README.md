# ENEM Admissions Fairness

Code accompanying the final paper **"TODO: paper title"**.

The paper asks whether machine-learning admissions systems **reproduce or exacerbate demographic inequalities** relative to a benchmark selection process. It uses the 2019 ENEM microdata published by INEP, Brazil's national education statistics institute.

## Overview

Machine-learning models (logistic regression and gradient boosting) are trained on ENEM 2019 data. Their scores are turned into admission decisions with a top-K rule, which selects the same share of candidates as the benchmark. Selection outcomes are then compared across demographic groups, including race, income, sex and school type.

A model is said to **practically reproduce** the benchmark when the difference in outcomes between the two, |D|, is at most 1 percentage point. This threshold is the **reproduction margin**. Robustness checks vary the selection share K, the model class and capacity, and how the composite score is constructed.

The income-heterogeneity analysis in §6.5 of the paper is exploratory. It was not pre-specified, and its results should be read as hypothesis-generating.

## Repository structure

```
DADOS/                          ENEM 2019 data folder (large CSVs not included, see "Data")
  sample_enem.py                Builds the analysis sample from the microdata   <!-- TODO: confirm -->
DICIONARIO/                     INEP data dictionary (.ods, .xlsx)
INPUTS/                         INEP's official loading scripts for R, SAS and SPSS
LEIA-ME E DOCUMENTOS TÉCNICOS/  INEP documentation (Edital, Leia-Me, reference matrix, analysis procedures)
PROVAS E GABARITOS/             2019 test booklets and answer keys (PDF)
TODO: list your own analysis scripts and output folders here
```

The `DICIONARIO`, `INPUTS`, `LEIA-ME E DOCUMENTOS TÉCNICOS` and `PROVAS E GABARITOS` folders are INEP's original microdata package, unmodified. They are included so that variable definitions can be checked alongside the code.

## Data

The following files are **not included**, because they exceed GitHub's 100 MB file limit:

- `DADOS/MICRODADOS_ENEM_2019.csv`, the full ENEM 2019 microdata
- `DADOS/enem_2019_sample.csv`, the analysis sample derived from it

To obtain them:

1. Download the ENEM 2019 microdata from INEP: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
2. Unzip the archive and copy `MICRODADOS_ENEM_2019.csv` into the `DADOS/` folder of this repository.
3. Generate the analysis sample by running the scripts below.

The microdata are public and anonymised by INEP.

Data accessed on: TODO (use the same date as Appendix D of the paper)

## How to reproduce

1. Clone this repository.
2. Install Python TODO (version) and the required packages:
   ```
   pip install TODO   # e.g. pandas numpy scikit-learn
   ```
3. Place the microdata in `DADOS/` as described above.
4. Run the scripts in order:
   ```
   python DADOS/sample_enem.py      # creates DADOS/enem_2019_sample.csv   (TODO: confirm)
   TODO: next scripts
   ```

## Outputs

The scripts produce the tables reported in the paper, including `robust_income_strata.csv`, the income-tercile robustness table for §6.5.

TODO: note where output files are saved.

## Use of AI assistance

Much of the code in this repository was drafted with the help of Claude, an AI assistant made by Anthropic, under the author's direction. The author designed the analysis, and reviewed, ran and checked all code, and is responsible for the results. See the declaration on assistive tools in Appendix D of the paper.

## Author

TODO: name, course, university, supervisor

## License

This repository is shared for academic assessment only. The ENEM microdata and documentation belong to INEP and are distributed under its open-data terms.
