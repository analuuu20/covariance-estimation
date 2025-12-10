#!/usr/bin/env python3
"""
main.py

Orchestrates the full pipeline for covariance estimation:
  1) Data ingestion & cleaning (01_data_download)
  2) Chronological panel split (02_data_split)
  3) Log-return calculation (03_returns_calculation)
  4) Baseline sample covariance (04_baseline_matrix)
  5) Train Graphical Lasso (05_graphical_lasso_training)
  6) Train Ledoit-Wolf (06_ledoit_wolf_training)
  7) Train DCC-GARCH (07_dcc_garch_training)
  8) Baseline validation computation (08_baseline_validation)
  9) Full validation & consolidation (09_model_validation)

Important design decisions:
 - Modules are loaded from `src/covariance_estimation/` using importlib,
   to allow filenames starting with digits (which are not valid identifiers).
 - Each high-level step prints concise, informative progress messages.
 - The script tries to reuse function return values where available, but
   also respects each module's own I/O (many modules read/write standard CSVs).
"""

from __future__ import annotations
import os
import sys
import time
import json
import importlib.util
from types import ModuleType
from typing import Dict, Any

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src", "covariance_estimation")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

# Map the expected module filenames to short keys and the primary function we want to call
MODULE_SPECS = [
    {"file": "01_data_download.py", "key": "data_download", "func": "full_download"},
    {"file": "02_data_split.py", "key": "data_split", "func": "chronological_panel_split"},
    {"file": "03_returns_calculation.py", "key": "returns_calc", "func": "compute_log_returns"},
    {"file": "04_baseline_matrix.py", "key": "baseline_matrix", "func": "baseline_training"},
    {"file": "05_graphical_lasso_training.py", "key": "glasso", "func": "graphical_lasso_training"},
    {"file": "06_ledoit_wolf_training.py", "key": "ledoit", "func": "ledoit_wolf_training"},
    {"file": "07_dcc_garch_training.py", "key": "dcc", "func": "dcc_garch_training"},
    {"file": "08_baseline_validation.py", "key": "baseline_validation", "func": "baseline_validation"},
    {"file": "09_model_validation.py", "key": "full_validation", "func": "validation"},
]


# --------------------------------------------------------------------
# Helper: dynamic module loader for filenames that are not valid module identifiers
# --------------------------------------------------------------------
def load_module_from_file(path: str, module_name: str) -> ModuleType:
    """
    Load a Python module from a file path. Returns the loaded module object.

    This uses importlib.util.spec_from_file_location so we can import modules whose
    filenames begin with digits (which means normal import syntax won't work).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SRC_DIR, exist_ok=True)  # safe-guard


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_run_summary(summary: Dict[str, Any], outpath: str):
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------
def main():
    print("\n[MAIN] Starting full pipeline")
    start_all = time.time()
    ensure_dirs()

    loaded_modules: Dict[str, ModuleType] = {}
    run_summary: Dict[str, Any] = {"started_at": timestamp(), "steps": {}}

    # 1) Load modules dynamically and keep references
    print("[MAIN] Loading modules from:", SRC_DIR)
    for spec in MODULE_SPECS:
        fpath = os.path.join(SRC_DIR, spec["file"])
        key = spec["key"]
        try:
            mod = load_module_from_file(fpath, f"covmod_{key}")
            loaded_modules[key] = mod
            print(f"[MAIN] Loaded module '{spec['file']}' -> key='{key}'")
        except Exception as e:
            print(f"[ERROR] Could not load module {spec['file']}: {e}")
            raise

    # -------------------------------
    # Step A — Data ingestion (01)
    # -------------------------------
    try:
        print("\n[STEP 01] Data download & cleaning (01_data_download)")
        t0 = time.time()
        mod = loaded_modules["data_download"]
        # call full_download and expect it to save and return cleaned DataFrame (as implemented)
        if hasattr(mod, "full_download"):
            df_clean = mod.full_download(output_path=os.path.join(DATA_DIR, "sp500_prices_clean.csv"))
        else:
            raise AttributeError("Module does not expose 'full_download' function.")
        elapsed = time.time() - t0
        run_summary["steps"]["01_data_download"] = {"status": "ok", "elapsed_s": elapsed}
        print(f"[STEP 01] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 01 failed:", e)
        run_summary["steps"]["01_data_download"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step B — Chronological split (02)
    # -------------------------------
    try:
        print("\n[STEP 02] Chronological panel split (02_data_split)")
        t0 = time.time()
        mod = loaded_modules["data_split"]

        # If the module exposes chronological_panel_split we use it; otherwise fall back to running the module main behavior
        if hasattr(mod, "chronological_panel_split"):
            # use the dataframe returned earlier (df_clean) if available
            train_df, val_df = mod.chronological_panel_split(df_clean, train_ratio=0.8)
            # Save to CSV (module main also does this normally)
            os.makedirs(DATA_DIR, exist_ok=True)
            train_path = os.path.join(DATA_DIR, "train_prices.csv")
            val_path = os.path.join(DATA_DIR, "validation_prices.csv")
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            # run validation checks (optional, for diagnostics)
            if hasattr(mod, "validate_split"):
                mod.validate_split(train_df, val_df)
        else:
            raise AttributeError("Module does not expose 'chronological_panel_split' function.")
        elapsed = time.time() - t0
        run_summary["steps"]["02_data_split"] = {"status": "ok", "elapsed_s": elapsed}
        print(f"[STEP 02] Completed in {elapsed:.1f}s — outputs saved to data/train_prices.csv and data/validation_prices.csv")
    except Exception as e:
        print("[FATAL] Step 02 failed:", e)
        run_summary["steps"]["02_data_split"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step C — Compute log-returns (03)
    # -------------------------------
    try:
        print("\n[STEP 03] Compute log-returns (03_returns_calculation)")
        t0 = time.time()
        mod = loaded_modules["returns_calc"]

        # load previously saved price files if needed (we have train_df & val_df in memory)
        if hasattr(mod, "compute_log_returns"):
            train_returns = mod.compute_log_returns(train_df)
            val_returns = mod.compute_log_returns(val_df)

            # save
            train_rpath = os.path.join(DATA_DIR, "train_returns.csv")
            val_rpath = os.path.join(DATA_DIR, "validation_returns.csv")
            train_returns.to_csv(train_rpath, index=False)
            val_returns.to_csv(val_rpath, index=False)

            # optional validation checks
            if hasattr(mod, "validate_returns_dataset"):
                mod.validate_returns_dataset(train_returns, "Training Set")
                mod.validate_returns_dataset(val_returns, "Validation Set")
        else:
            raise AttributeError("Module does not expose 'compute_log_returns' function.")

        elapsed = time.time() - t0
        run_summary["steps"]["03_returns_calculation"] = {"status": "ok", "elapsed_s": elapsed}
        print(f"[STEP 03] Completed in {elapsed:.1f}s — outputs saved to {train_rpath} and {val_rpath}")
    except Exception as e:
        print("[FATAL] Step 03 failed:", e)
        run_summary["steps"]["03_returns_calculation"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step D — Baseline covariance from training (04)
    # -------------------------------
    try:
        print("\n[STEP 04] Baseline sample covariance (04_baseline_matrix)")
        t0 = time.time()
        mod = loaded_modules["baseline_matrix"]
        # baseline_training reads data/train_returns.csv and writes results/training/baseline/...
        if hasattr(mod, "baseline_training"):
            mod.baseline_training()
        else:
            raise AttributeError("Module does not expose 'baseline_training'")
        elapsed = time.time() - t0
        run_summary["steps"]["04_baseline_matrix"] = {"status": "ok", "elapsed_s": elapsed}
        print(f"[STEP 04] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 04 failed:", e)
        run_summary["steps"]["04_baseline_matrix"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step E — Graphical Lasso training (05)
    # -------------------------------
    try:
        print("\n[STEP 05] Graphical Lasso training (05_graphical_lasso_training)")
        t0 = time.time()
        mod = loaded_modules["glasso"]
        if hasattr(mod, "graphical_lasso_training"):
            gl_model, gl_cov, gl_prec, gl_tickers = mod.graphical_lasso_training()
            run_summary["steps"]["05_graphical_lasso"] = {
                "status": "ok",
                "n_tickers": len(gl_tickers)
            }
        else:
            raise AttributeError("Module does not expose 'graphical_lasso_training'")
        elapsed = time.time() - t0
        print(f"[STEP 05] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 05 failed:", e)
        run_summary["steps"]["05_graphical_lasso"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step F — Ledoit-Wolf training (06)
    # -------------------------------
    try:
        print("\n[STEP 06] Ledoit-Wolf training (06_ledoit_wolf_training)")
        t0 = time.time()
        mod = loaded_modules["ledoit"]
        if hasattr(mod, "ledoit_wolf_training"):
            lw_model, lw_cov, lw_tickers = mod.ledoit_wolf_training()
            run_summary["steps"]["06_ledoit_wolf"] = {
                "status": "ok",
                "n_tickers": len(lw_tickers)
            }
        else:
            raise AttributeError("Module does not expose 'ledoit_wolf_training'")
        elapsed = time.time() - t0
        print(f"[STEP 06] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 06 failed:", e)
        run_summary["steps"]["06_ledoit_wolf"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step G — DCC-GARCH training (07)
    # -------------------------------
    try:
        print("\n[STEP 07] DCC-GARCH training (07_dcc_garch_training)")
        t0 = time.time()
        mod = loaded_modules["dcc"]
        if hasattr(mod, "dcc_garch_training"):
            # call with defaults - module itself tries to align to canonical tickers (491) if available
            dcc_results = mod.dcc_garch_training()
            run_summary["steps"]["07_dcc_garch"] = {
                "status": "ok",
                "tickers_kept": len(dcc_results.get("tickers_kept", []))
            }
        else:
            raise AttributeError("Module does not expose 'dcc_garch_training'")
        elapsed = time.time() - t0
        print(f"[STEP 07] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 07 failed:", e)
        run_summary["steps"]["07_dcc_garch"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step H — Baseline validation covariance (08)
    # -------------------------------
    try:
        print("\n[STEP 08] Baseline validation covariance computation (08_baseline_validation)")
        t0 = time.time()
        mod = loaded_modules["baseline_validation"]
        if hasattr(mod, "baseline_validation"):
            mod.baseline_validation()
            run_summary["steps"]["08_baseline_validation"] = {"status": "ok"}
        else:
            raise AttributeError("Module does not expose 'baseline_validation'")
        elapsed = time.time() - t0
        print(f"[STEP 08] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 08 failed:", e)
        run_summary["steps"]["08_baseline_validation"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Step I — Full validation & consolidation (09)
    # -------------------------------
    try:
        print("\n[STEP 09] Full validation & consolidation (09_model_validation)")
        t0 = time.time()
        mod = loaded_modules["full_validation"]
        if hasattr(mod, "validation"):
            results = mod.validation()
            run_summary["steps"]["09_model_validation"] = {"status": "ok"}
            # optionally include a summary of consolidated ranking if available
            if isinstance(results, dict) and "consolidated" in results:
                try:
                    top = results["consolidated"].sort_values("score").head()
                    run_summary["steps"]["09_model_validation"]["consolidated_head"] = top.to_dict(orient="records")
                except Exception:
                    pass
        else:
            raise AttributeError("Module does not expose 'validation'")
        elapsed = time.time() - t0
        print(f"[STEP 09] Completed in {elapsed:.1f}s")
    except Exception as e:
        print("[FATAL] Step 09 failed:", e)
        run_summary["steps"]["09_model_validation"] = {"status": "failed", "error": str(e)}
        raise

    # -------------------------------
    # Finalize & save run summary
    # -------------------------------
    total_elapsed = time.time() - start_all
    run_summary["finished_at"] = timestamp()
    run_summary["total_elapsed_s"] = total_elapsed
    print("\n[MAIN] Pipeline finished in {:.1f}s".format(total_elapsed))

    summary_path = os.path.join(RESULTS_DIR, "pipeline_run_summary.json")
    try:
        save_run_summary(run_summary, summary_path)
        print(f"[MAIN] Run summary saved: {summary_path}")
    except Exception as e:
        print("[WARN] Could not save run summary:", e)

    # Print final consolidated ranking summary if available in run_summary
    if "09_model_validation" in run_summary.get("steps", {}):
        print("[MAIN] Full validation step completed; check results/validation for details.")

    print("[MAIN] Done.\n")
    return run_summary


if __name__ == "__main__":
    main()

