"""Mixed-effects analysis from the model-input table.

This module assumes that the model-input table has already been
created. It starts from a compact table with columns such as ``creep_at_10s``,
``stage``, ``individual_id``, ``probe_size``, ``x_position``, and
``y_position`` and fits the linear mixed-effects model used for the
manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import build_design_matrices


DEFAULT_STAGE_ORDER = (
  "1 cell stage",
  "2 cell stage",
  "4 cell stage",
  "8 cell stage",
)

DEFAULT_STAGE_COLORS = {
  "1 cell stage": "red",
  "2 cell stage": "blue",
  "4 cell stage": "green",
  "8 cell stage": "purple",
}

DEFAULT_COLUMN_MAP = {
  "y": "creep_at_10s",
  "stage": "stage",
  "individual": "individual_id",
  "probe_size": "probe_size",
  "x_position": "x_position",
  "y_position": "y_position",
}


@dataclass(frozen=True)
class MixedLMConfig:
  """Configuration for fitting the mixed-effects model from input data."""

  y_col: str = "creep_at_10s"
  stage_col: str = "stage"
  individual_col: str = "individual_id"
  probe_size_col: str = "probe_size"
  x_position_col: str = "x_position"
  y_position_col: str = "y_position"
  stage_order: Sequence[str] = DEFAULT_STAGE_ORDER
  reference_stage: str = "1 cell stage"
  stage_colors: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_STAGE_COLORS))
  standardize_x: bool = True
  reml: bool = True
  maxiter: int = 2000


@dataclass
class MixedLMAnalysisResult:
  """Container returned by :func:`run_mixedlm_analysis`."""

  model_result: object
  data_used: pd.DataFrame
  fixed_effects: pd.DataFrame
  metadata: dict


def load_model_input_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
  """Load the model-input table from Excel or CSV."""
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"Input file was not found: {path}")

  suffix = path.suffix.lower()
  if suffix in {".xlsx", ".xls"}:
    return pd.read_excel(path, sheet_name=sheet_name)
  if suffix == ".csv":
    return pd.read_csv(path)
  raise ValueError(f"Unsupported input file type: {suffix}. Use .xlsx, .xls, or .csv.")


def validate_model_input_table(df: pd.DataFrame, config: MixedLMConfig = MixedLMConfig()) -> None:
  """Validate that all required columns are present in the input table."""
  required = [
    config.y_col,
    config.stage_col,
    config.individual_col,
    config.probe_size_col,
    config.x_position_col,
    config.y_position_col,
  ]
  missing = [col for col in required if col not in df.columns]
  if missing:
    raise KeyError(f"The input table is missing required columns: {missing}")


def _zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
  values = pd.to_numeric(series, errors="coerce")
  mu = float(values.mean())
  sd = float(values.std(ddof=0))
  if not np.isfinite(sd) or sd == 0:
    return values * np.nan, mu, sd
  return (values - mu) / sd, mu, sd


def prepare_mixedlm_dataframe(
  df: pd.DataFrame,
  config: MixedLMConfig = MixedLMConfig(),
  dropna: bool = True,
) -> tuple[pd.DataFrame, dict]:
  """Prepare the design dataframe for the mixed-effects model.

  The probe-size variable is summarized as an embryo-level mean. The position
  variables are treated as observation-level covariates. Continuous predictors
  are standardized by default, so their coefficients are interpreted per 1 SD.
  """
  validate_model_input_table(df, config)

  d = df.copy()
  d.columns = [str(c).strip() for c in d.columns]

  numeric_cols = [
    config.y_col,
    config.probe_size_col,
    config.x_position_col,
    config.y_position_col,
  ]
  for col in numeric_cols:
    d[col] = pd.to_numeric(d[col], errors="coerce")

  d[config.individual_col] = d[config.individual_col].astype(str)
  d[config.stage_col] = pd.Categorical(
    d[config.stage_col].astype(str),
    categories=list(config.stage_order),
    ordered=True,
  )

  probe_mean_col = f"{config.probe_size_col}_individual_mean"
  d[probe_mean_col] = d.groupby(config.individual_col)[config.probe_size_col].transform("mean")

  raw_predictors = [probe_mean_col, config.x_position_col, config.y_position_col]
  x_mu: dict[str, float] = {}
  x_sd: dict[str, float] = {}

  if config.standardize_x:
    model_predictors = []
    for col in raw_predictors:
      z_col = f"z_{col}"
      d[z_col], x_mu[col], x_sd[col] = _zscore(d[col])
      model_predictors.append(z_col)
  else:
    model_predictors = raw_predictors
    for col in raw_predictors:
      values = pd.to_numeric(d[col], errors="coerce")
      x_mu[col] = float(values.mean())
      x_sd[col] = float(values.std(ddof=0))

  required = [config.y_col, config.individual_col, config.stage_col, *model_predictors]
  if dropna:
    d = d.dropna(subset=required).copy()

  d[config.stage_col] = d[config.stage_col].cat.remove_unused_categories()
  if len(d) == 0:
    raise ValueError("No rows remain after removing missing model variables.")

  metadata = {
    "raw_predictors": raw_predictors,
    "model_predictors": model_predictors,
    "probe_mean_col": probe_mean_col,
    "x_mu": x_mu,
    "x_sd": x_sd,
    "stage_order": list(config.stage_order),
    "reference_stage": config.reference_stage,
    "standardize_x": config.standardize_x,
  }
  return d, metadata


def fit_mixedlm_model(
  df: pd.DataFrame,
  config: MixedLMConfig = MixedLMConfig(),
):
  """Fit the linear mixed-effects model.

  Model structure:
  - response: creep at 10 seconds
  - fixed effects: stage, probe size, X position, and Y position
  - random effect: embryo/individual random intercept
  """
  d, metadata = prepare_mixedlm_dataframe(df, config=config, dropna=True)
  predictors = metadata["model_predictors"]
  ref = config.reference_stage

  formula = (
    f"{config.y_col} ~ "
    + " + ".join(predictors)
    + f' + C({config.stage_col}, Treatment(reference="{ref}"))'
  )

  model = smf.mixedlm(
    formula=formula,
    data=d,
    groups=d[config.individual_col],
    re_formula="1",
    missing="drop",
  )
  result = model.fit(reml=config.reml, method="lbfgs", maxiter=config.maxiter, disp=False)
  metadata["formula"] = formula
  metadata["n_rows_used"] = int(result.nobs)
  metadata["n_individuals"] = int(d[config.individual_col].nunique())
  return result, d, metadata


def fixed_effects_table(model_result, metadata: dict) -> pd.DataFrame:
  """Return a publication-friendly fixed-effect coefficient table."""
  fe = model_result.fe_params.copy()
  bse = getattr(model_result, "bse_fe", model_result.bse).reindex(fe.index)
  z_values = fe / bse.replace(0, np.nan)

  from math import erf, sqrt

  def p_from_z(z):
    if not np.isfinite(z):
      return np.nan
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))

  p_values = z_values.apply(p_from_z)

  ci = model_result.conf_int().reindex(fe.index)
  ci.columns = ["ci_lower", "ci_upper"]

  table = pd.DataFrame(
    {
      "term": fe.index,
      "estimate": fe.values,
      "std_error": bse.values,
      "z_value": z_values.values,
      "p_value": p_values.values,
      "ci_lower": ci["ci_lower"].values,
      "ci_upper": ci["ci_upper"].values,
    }
  )

  x_sd = metadata.get("x_sd", {})
  table["predictor_scale"] = ""
  table["estimate_per_original_unit"] = np.nan
  for i, term in table["term"].items():
    if isinstance(term, str) and term.startswith("z_"):
      raw = term[2:]
      sd = x_sd.get(raw, np.nan)
      table.loc[i, "predictor_scale"] = "per 1 SD"
      if np.isfinite(sd) and sd != 0:
        table.loc[i, "estimate_per_original_unit"] = table.loc[i, "estimate"] / sd
    elif term == "Intercept" or "C(" in str(term):
      table.loc[i, "predictor_scale"] = "stage contrast"
  return table


def save_model_outputs(
  model_result,
  data_used: pd.DataFrame,
  fixed_effects: pd.DataFrame,
  metadata: dict,
  outdir: str | Path,
  prefix: str = "mixedlm",
) -> None:
  """Save model summary, coefficients, metadata, and the fitted dataset."""
  outdir = Path(outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  fixed_effects.to_csv(outdir / f"{prefix}_fixed_effects.csv", index=False, encoding="utf-8-sig")
  data_used.to_csv(outdir / f"{prefix}_data_used.csv", index=False, encoding="utf-8-sig")
  with open(outdir / f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
    f.write(model_result.summary().as_text())
  pd.Series({k: str(v) for k, v in metadata.items()}).to_csv(
    outdir / f"{prefix}_metadata.csv", header=["value"], encoding="utf-8-sig"
  )


def run_mixedlm_analysis(
  input_path: str | Path,
  outdir: str | Path = "mixedlm_outputs",
  config: MixedLMConfig = MixedLMConfig(),
  save_outputs: bool = True,
  prefix: str = "mixedlm",
) -> MixedLMAnalysisResult:
  """Load the model-input table, fit the mixed-effects model, and optionally save outputs."""
  df = load_model_input_table(input_path)
  model_result, data_used, metadata = fit_mixedlm_model(df, config=config)
  fe_table = fixed_effects_table(model_result, metadata)
  if save_outputs:
    save_model_outputs(model_result, data_used, fe_table, metadata, outdir=outdir, prefix=prefix)
  return MixedLMAnalysisResult(model_result, data_used, fe_table, metadata)


def _ensure_outdir(outdir: str | Path) -> Path:
  outdir = Path(outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir


def _prediction_dataframe(model_result, new_data: pd.DataFrame) -> pd.DataFrame:
  design_info = model_result.model.data.design_info
  return build_design_matrices([design_info], new_data, return_type="dataframe")[0]


def add_fixed_effect_predictions(model_result, data: pd.DataFrame) -> pd.DataFrame:
  """Add fixed-effect predictions and residuals to the fitted dataset."""
  d = data.copy()
  exog = _prediction_dataframe(model_result, d)
  pred = np.asarray(exog @ model_result.fe_params.reindex(exog.columns).values, dtype=float)
  d["predicted_fixed_effect"] = pred
  d["residual_fixed_effect"] = d[model_result.model.endog_names].to_numpy(dtype=float) - pred
  return d


def plot_observed_vs_predicted(
  model_result,
  data_used: pd.DataFrame,
  outdir: str | Path = "mixedlm_figures",
  filename: str = "observed_vs_predicted",
  show: bool = False,
) -> None:
  """Plot observed creep values against fixed-effect predictions."""
  outdir = _ensure_outdir(outdir)
  d = add_fixed_effect_predictions(model_result, data_used)
  y_col = model_result.model.endog_names

  fig, ax = plt.subplots(figsize=(4.2, 4.0))
  ax.scatter(d["predicted_fixed_effect"], d[y_col], s=28, alpha=0.8)
  low = np.nanmin([d["predicted_fixed_effect"].min(), d[y_col].min()])
  high = np.nanmax([d["predicted_fixed_effect"].max(), d[y_col].max()])
  ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
  ax.set_xlabel("Predicted creep at 10 s")
  ax.set_ylabel("Observed creep at 10 s")
  ax.set_title("Observed vs predicted")
  fig.tight_layout()
  fig.savefig(outdir / f"{filename}.png", dpi=300, bbox_inches="tight")
  fig.savefig(outdir / f"{filename}.pdf", dpi=300, bbox_inches="tight")
  if show:
    plt.show()
  plt.close(fig)


def plot_stage_estimates(
  model_result,
  config: MixedLMConfig = MixedLMConfig(),
  outdir: str | Path = "mixedlm_figures",
  filename: str = "stage_estimates",
  show: bool = False,
) -> None:
  """Plot estimated stage-specific intercepts with approximate 95% CIs."""
  outdir = _ensure_outdir(outdir)
  fe = model_result.fe_params
  cov = model_result.cov_params().loc[fe.index, fe.index]
  intercept = "Intercept"

  rows = []
  for stage in config.stage_order:
    if stage == config.reference_stage:
      estimate = float(fe[intercept])
      variance = float(cov.loc[intercept, intercept])
    else:
      term = next((t for t in fe.index if f"[T.{stage}]" in t), None)
      if term is None:
        rows.append((stage, np.nan, np.nan))
        continue
      estimate = float(fe[intercept] + fe[term])
      variance = float(cov.loc[intercept, intercept] + cov.loc[term, term] + 2 * cov.loc[intercept, term])
    se = np.sqrt(variance) if variance >= 0 else np.nan
    rows.append((stage, estimate, se))

  stage_df = pd.DataFrame(rows, columns=["stage", "estimate", "se"])
  stage_df["ci_lower"] = stage_df["estimate"] - 1.96 * stage_df["se"]
  stage_df["ci_upper"] = stage_df["estimate"] + 1.96 * stage_df["se"]
  stage_df.to_csv(outdir / f"{filename}.csv", index=False, encoding="utf-8-sig")

  x = np.arange(len(stage_df))
  fig, ax = plt.subplots(figsize=(5.2, 4.0))
  for i, row in stage_df.iterrows():
    color = config.stage_colors.get(row["stage"], "black")
    yerr = [[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]]
    ax.errorbar(i, row["estimate"], yerr=yerr, fmt="o", capsize=3, color=color)
  ax.set_xticks(x)
  ax.set_xticklabels(stage_df["stage"], rotation=20, ha="right")
  ax.set_ylabel("Estimated creep at 10 s")
  ax.set_xlabel("Stage")
  fig.tight_layout()
  fig.savefig(outdir / f"{filename}.png", dpi=300, bbox_inches="tight")
  fig.savefig(outdir / f"{filename}.pdf", dpi=300, bbox_inches="tight")
  if show:
    plt.show()
  plt.close(fig)


def plot_covariate_effects(
  model_result,
  data_used: pd.DataFrame,
  metadata: dict,
  config: MixedLMConfig = MixedLMConfig(),
  outdir: str | Path = "mixedlm_figures",
  prefix: str = "covariate_effect",
  show: bool = False,
) -> None:
  """Plot model-predicted covariate effects while holding other variables fixed."""
  outdir = _ensure_outdir(outdir)
  raw_predictors = metadata["raw_predictors"]
  model_predictors = metadata["model_predictors"]
  ref_stage = config.reference_stage

  base = data_used.iloc[[0]].copy()
  base[config.stage_col] = ref_stage
  for col in model_predictors:
    base[col] = 0.0

  for raw, term in zip(raw_predictors, model_predictors):
    xs = np.linspace(data_used[raw].min(), data_used[raw].max(), 80)
    plot_df = pd.concat([base] * len(xs), ignore_index=True)
    if config.standardize_x:
      mu = metadata["x_mu"][raw]
      sd = metadata["x_sd"][raw]
      plot_df[term] = (xs - mu) / sd if sd != 0 else 0.0
    else:
      plot_df[term] = xs

    exog = _prediction_dataframe(model_result, plot_df)
    pred = np.asarray(exog @ model_result.fe_params.reindex(exog.columns).values, dtype=float)

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot(xs, pred, linewidth=1.8)
    ax.set_xlabel(raw.replace("_individual_mean", "").replace("_", " "))
    ax.set_ylabel("Predicted creep at 10 s")
    ax.set_title(f"Effect of {raw}")
    fig.tight_layout()
    safe_name = raw.replace("/", "_")
    fig.savefig(outdir / f"{prefix}_{safe_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{prefix}_{safe_name}.pdf", dpi=300, bbox_inches="tight")
    if show:
      plt.show()
    plt.close(fig)


def make_standard_diagnostic_plots(
  analysis: MixedLMAnalysisResult,
  outdir: str | Path = "mixedlm_figures",
  config: MixedLMConfig = MixedLMConfig(),
  show: bool = False,
) -> None:
  """Generate the standard diagnostic and publication-support figures."""
  plot_stage_estimates(analysis.model_result, config=config, outdir=outdir, show=show)
  plot_observed_vs_predicted(analysis.model_result, analysis.data_used, outdir=outdir, show=show)
  plot_covariate_effects(
    analysis.model_result,
    analysis.data_used,
    analysis.metadata,
    config=config,
    outdir=outdir,
    show=show,
  )
