from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

# PLOTTING FUNCTIONS

def plot_fano_traces(
        all_mice: dict,
        colour_lib: dict,
    ) -> list:
  
    out = []

    for midx, keys in all_mice.items():

        keys["Familiar"].tpc_ts
  
        # DEFINING PLOT

        fig, axes = plt.subplots(
            1, 5,
            figsize=(11.69, 2.75),
            dpi=300,
            gridspec_kw={
            "width_ratios": [1,1,1,1,0.32],
            "wspace": 0.12,
            },
            sharex=True,
            sharey=True
        )

        # ASSIGNING GRID FOR REFERENCE

        panel_map = {
                (True,  True):  0,  # active, familiar
                (True,  False): 1,  # active, novel
                (False, True):  2,  # passive, familiar
                (False, False): 3,  # passive, novel
            }
        
        # COLLECTING TRACES ACROSS MOUSE

        i = 0

        for key, dat in keys['Familiar'].fano.items():
            print(key)
            print(dat).fano
            if i >= 1:
                return None
            i += 1

        familiar_traces = {key: dat['fano_mm'] for key, dat in keys['Familiar'].fano.items()}
        novel_traces = {key: dat['fano_mm'] for key, dat in keys['Novel'].fano.items()}

        combined = {
            (*k, True): v for k, v in familiar_traces.items()
        } | {
            (*k, False): v for k, v in novel_traces.items()
        }

        # PLOTTING TRACES AND COLLECTING TRACES ACROSS ALL CONDITIONS

        pool_all = []

        for (area, active, familiar), trace in combined.items():

            ax = axes[panel_map[(active, familiar)]]
            ax.plot(tpc_ts, trace, color=colour_lib[area], lw=1.2, label=str(area))
            pool_all.append(trace)

        # FINDING GLOBAL MIN AND MAX AND SETTING LIMITS TO AXIS

        stack = np.stack(pool_all, axis=0)
        ymin = np.floor(np.nanmin(stack) * 10) / 10
        ymax = np.ceil(np.nanmax(stack) * 10) / 10
        ticks = np.arange(ymin, ymax + 0.1 / 2, 0.1)

        # FORMATTING PANELS

        for (active, familiar), idx in panel_map.items():

            ax = axes[idx]

            ax.axvline(0, color="k", alpha=0.2, linestyle="dashed", zorder=0)
            ax.axvline(0.25, color="k", alpha=0.2, linestyle="dashed", zorder=0)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            ax.tick_params(axis="x", bottom=False, labelbottom=False)

            if idx == 0:
                ax.spines["left"].set_position(("outward", 6))
                ax.spines["left"].set_capstyle("round")
                ax.spines["left"].set_linewidth(1.1)

                ax.set_xticks([0.0, 0.25])
                ax.set_xticklabels([r"$t=0$", r"$t=0.25$"])

                ax.tick_params(
                    axis="x",
                    bottom=True,
                    labelbottom=True,
                    length=0,
                    width=0,
                    labelsize=8,
                )

                ax.tick_params(
                    axis="y",
                    direction="out",
                    length=3,
                    width=1.1,
                    labelsize=8,
                    right=False,
                )
                ax.set_ylabel("Fano Factor", fontsize=9, labelpad=6)
            else:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)

                ax.set_ylim(ymin, ymax)
                ax.set_yticks(ticks)

                cc = "active" if active else "passive"
                ff = "familiar" if familiar else "novel"
                ax.set_title(f"{cc} × {ff}", fontsize=9, pad=4)

                leg_ax = axes[-1]
                leg_ax.axis("off")

                handles, labels = axes[0].get_legend_handles_labels()
                leg_ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    frameon=False,
                    fontsize=8,
                    handlelength=2.2,
                    labelspacing=0.35,
                )

        fig.subplots_adjust(left=0.045, right=0.995, top=0.88, bottom=0.12)
        fig.suptitle(f"Fano Factor traces for mouse {midx}")

        out.append(fig)

    return out

