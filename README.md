# Mixed-effects analysis from model-input data

This repository contains Python code for the mixed-effects model analysis used in the following study:
Koizumi, S., Tokuyasu, A., Miyamoto, A.W.M., Torisawa, T., Tanimoto, T., and Kimura, A. (2026). Developmentally programmed changes in cytoplasmic mechanics revealed by active microrheology in C. elegans embryos. bioRxiv. https://doi.org/10.64898/2026.05.19.726147

The analysis starts from a curated model-input table rather than from raw tracking data. Both a Python script and a Jupyter notebook are provided to facilitate inspection and reproduction of the analysis workflow.

## Files

```text
mixedlm_analysis.py                    # Compact analysis functions
model_input_data.xlsx                  # Example/model-input data table
examples/run_mixedlm_analysis.py       # Script usage example
examples/mixedlm_analysis_example.ipynb # Notebook usage example
requirements.txt                       # Python dependencies
```

## Input table

The expected columns are:

| Column | Meaning |
|---|---|
| `creep_at_10s` | response variable, creep at 10 seconds |
| `stage` | cell stage |
| `individual_id` | embryo/individual ID |
| `probe_size` | probe size |
| `x_position` | X position |
| `y_position` | Y position |

## Installation

Install the required Python packages with:
```bash
pip install -r requirements.txt
```

## Usage

Run the example analysis with:
```bash
python examples/run_mixedlm_analysis.py
```

Alternatively, open and run:
```text
examples/mixedlm_analysis_example.ipynb
```

## Statistical Model

- Response: `creep_at_10s`
- Fixed effects: stage, probe size, X position, Y position
- Random effect: embryo/individual random intercept
- Continuous predictors are standardized by default, so their coefficients are per 1 SD.


## License

This code is distributed under the MIT License. See LICENSE for details.