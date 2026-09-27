"""Example: fit the mixed-effects model from the model-input data table."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mixedlm_analysis import (  # noqa: E402
    MixedLMConfig,
    make_standard_diagnostic_plots,
    run_mixedlm_analysis,
)

INPUT_PATH = REPO_ROOT / "model_input_data.xlsx"
OUTPUT_DIR = REPO_ROOT / "mixedlm_outputs"
FIGURE_DIR = REPO_ROOT / "mixedlm_figures"

config = MixedLMConfig(
    y_col="creep_at_10s",
    stage_col="stage",
    individual_col="individual_id",
    probe_size_col="probe_size",
    x_position_col="x_position",
    y_position_col="y_position",
)

analysis = run_mixedlm_analysis(
    input_path=INPUT_PATH,
    outdir=OUTPUT_DIR,
    config=config,
    save_outputs=True,
    prefix="mixedlm",
)

make_standard_diagnostic_plots(
    analysis,
    outdir=FIGURE_DIR,
    config=config,
    show=False,
)

print(analysis.model_result.summary())
print("Fixed-effect table:")
print(analysis.fixed_effects)
print(f"Outputs saved to: {Path(OUTPUT_DIR).resolve()}")
print(f"Figures saved to: {Path(FIGURE_DIR).resolve()}")
