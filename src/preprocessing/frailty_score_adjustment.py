import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import statsmodels.formula.api as smf

    from pathlib import Path
    from scipy import stats
    from typing import Dict, Any, Optional

    return Path, os, pd, plt, smf, stats


@app.cell
def _(Path, os):
    unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])
    datasets_dir = unsupervised_aging_dir / "data/datasets/"

    combined_1128_dir = datasets_dir / "combined_1128"
    geroscience_492_dir = datasets_dir / "geroscience_492"
    nature_aging_634_dir = datasets_dir / "nature-aging_634"
    return combined_1128_dir, geroscience_492_dir, nature_aging_634_dir


@app.cell
def _(plt, smf, stats):
    def frailty_score_adjustment(df):
        df_unique = df.drop_duplicates("mouse_id").copy()
        vc = {"batch": "0 + C(batch)"}
        if "diet" in df_unique.columns:
            vc["diet"] = "0 + C(diet)"
        full_md = smf.mixedlm(
            "fi ~ age + sex",
            df_unique,
            groups="tester",
            re_formula="1",
            vc_formula=vc,
        )
        full_fit = full_md.fit(reml=True)
        df_unique["_grp"] = 0
        red_md = smf.mixedlm(
            "fi ~ age + sex",
            df_unique,
            groups="_grp",
            vc_formula=vc,
        )
        print(full_md, red_md)
        red_fit = red_md.fit(reml=True)
        lr_stat = 2 * (full_fit.llf - red_fit.llf)
        p_val = stats.chi2.sf(lr_stat, 1)
        tester_re = {k: v.iloc[0] for k, v in full_fit.random_effects.items()}
        try:
            batch_re = {k.split(".")[1]: v for k, v in full_fit.vc_random_effects["batch"].items()}
        except AttributeError:
            tmp = df.copy()
            tmp["fi_adj_tester"] = tmp["fi"] - tmp["tester"].map(tester_re)
            batch_mean = tmp.groupby("batch")["fi_adj_tester"].mean()
            batch_re = (batch_mean - batch_mean.mean()).to_dict()
        if "diet" in df.columns:
            try:
                diet_re = {k.split(".")[1]: v for k, v in full_fit.vc_random_effects["diet"].items()}
            except AttributeError:
                tmp = df.copy()
                tmp["fi_adj_tb"] = tmp["fi"] - tmp["tester"].map(tester_re) - tmp["batch"].map(batch_re)
                diet_mean = tmp.groupby("diet")["fi_adj_tb"].mean()
                diet_re = (diet_mean - diet_mean.mean()).to_dict()
        else:
            diet_re = {}
        df_adj = df.copy()
        df_adj["fi"] = df_adj.apply(
            lambda r: max(r["fi"] - tester_re.get(r["tester"], 0) - batch_re.get(r["batch"], 0) - (diet_re.get(r["diet"], 0) if "diet" in df.columns else 0), 0),
            axis=1,
        )
        drop_cols = ["tester", "sex", "weight", "batch"]
        if "diet" in df.columns:
            drop_cols.append("diet")
        df_out = df_adj.drop(columns=drop_cols)
        df_out["__fi_raw"] = df["fi"]
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        ax_bar.bar(tester_re.keys(), tester_re.values())
        ax_bar.set_xlabel("Tester")
        ax_bar.set_ylabel("Estimated random effect")
        fig_bar.tight_layout()
        fig_scatter, ax_scatter = plt.subplots(figsize=(5, 5))
        ax_scatter.scatter(df_out["__fi_raw"], df_out["fi"], alpha=0.6)
        max_val = df_out[["__fi_raw", "fi"]].to_numpy().max()
        ax_scatter.plot([0, max_val], [0, max_val], ls="--", lw=1)
        ax_scatter.set_xlabel("Raw FI")
        ax_scatter.set_ylabel("Adjusted FI")
        ax_scatter.set_aspect("equal", adjustable="box")
        fig_scatter.tight_layout()
        tester_var = float(full_fit.cov_re.iloc[0, 0])
        vc_vals  = getattr(full_fit, "vcomp", [])
        vc_names = getattr(full_fit.model, "vc_names", list(vc.keys()))
        vcomp    = dict(zip(vc_names, vc_vals))
        batch_var = float(vcomp.get("batch", 0.0))
        diet_var  = float(vcomp.get("diet",  0.0))
        resid_var = float(full_fit.scale)
        tester_sd = tester_var ** 0.5
        batch_sd  = batch_var ** 0.5
        diet_sd   = diet_var ** 0.5
        resid_sd  = resid_var ** 0.5
        total_sd  = tester_sd + batch_sd + diet_sd + resid_sd
        var_prop = {
            "tester"  : tester_sd / total_sd if total_sd else 0,
            "batch"   : batch_sd  / total_sd if total_sd else 0,
            "diet"    : diet_sd   / total_sd if total_sd else 0,
            "residual": resid_sd  / total_sd if total_sd else 0,
        }
        df_out = df_out.drop("__fi_raw", axis=1)
        return df_out, fig_bar, fig_scatter, (lr_stat, p_val), var_prop

    return (frailty_score_adjustment,)


@app.cell
def _():
    # temp_df = pd.read_csv(combined_1128_dir / "metadata.csv")
    # _scorer_map = {
    #     "Scorer1": "Amanda",
    #     "Scorer2": "Gaven",
    #     "Scorer3": "Hannah",
    #     "Scorer4": "Mackenzie",
    # }

    # temp_df["tester"] = temp_df["tester"].replace(_scorer_map)
    # temp_df.to_csv(combined_1128_dir / "metadata.csv", index=False)
    return


@app.cell
def _(frailty_score_adjustment, nature_aging_634_dir, pd):
    nature_aging_634_df = pd.read_csv(nature_aging_634_dir / "metadata.csv")
    nature_aging_634_adj_df, _tester_effect, _raw_vs_adj, _lr_test, _var_prop = frailty_score_adjustment(nature_aging_634_df)
    print(f"(likelihood ratio test stat., χ² test p-value) = {_lr_test}")
    (
        _tester_effect,
        _raw_vs_adj,
        _var_prop,
        nature_aging_634_adj_df
    )
    return


@app.cell
def _(frailty_score_adjustment, geroscience_492_dir, pd):
    geroscience_492_df = pd.read_csv(geroscience_492_dir / "metadata.csv")
    geroscience_492_adj_df, _tester_effect, _raw_vs_adj, _lr_test, _var_prop = frailty_score_adjustment(geroscience_492_df)
    print(f"(likelihood ratio test stat., χ² test p-value) = {_lr_test}")
    (
        _tester_effect,
        _raw_vs_adj,
        _var_prop,
        geroscience_492_adj_df
    )
    return


@app.cell
def _(combined_1128_dir, frailty_score_adjustment, pd):
    combined_1128_df = pd.read_csv(combined_1128_dir / "metadata.csv")
    combined_1128_df, _tester_effect, _raw_vs_adj, _lr_test, _var_prop = frailty_score_adjustment(combined_1128_df)
    print(f"(likelihood ratio test stat., χ² test p-value) = {_lr_test}")
    (
        _tester_effect,
        _raw_vs_adj,
        _var_prop,
        combined_1128_df
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
