import numpy as np
from matplotlib import pyplot as plt


def pop_size_fig(df, project_root, fig_name, exp_name):
    fig, ax1 = plt.subplots()

    # Left axis
    ax1.plot(df["generation"], df["ne"], color="blue", label="Effective population size (Ne)")
    ax1.plot(df["generation"], df["nf"], color="red", label="Number of colonies (breeding queens)")
    ax1.plot(df["generation"], df["nm"], color="purple", label="Breeding males")
    ax1.plot(df["generation"], df["n"], color="orange", label="population size (N)")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population size (log10)")
    ax1.set_yscale("log")

    # Right axis
    ax2 = ax1.twinx()
    ax2.plot(df["generation"], df["expected_heterozygosity"],
             color="green", label="Expected heterozygosity")
    ax2.set_ylabel("Expected heterozygosity")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
    bbox_to_anchor=(1.12, 0.5))



    plt.savefig(project_root / "figures" / exp_name /fig_name,
                dpi=300, bbox_inches="tight")

def col_size_fig(df, project_root, fig_name, exp_name):
    fig, ax1 = plt.subplots()

    # Left axis
    #ax1.plot(df["generation"], df["ne"], color="blue", label="Effective population size (Ne)")

    ax1.plot(df["generation"], df["nf"], color="red", label="Number of colonies (breeding queens)")

    #ax1.plot(df["generation"], df["nm"], color="purple", label="Breeding males")
    #ax1.plot(df["generation"], df["n"], color="orange", label="population size (N)")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population size (log10)")
    ax1.set_yscale("log")

    # Right axis
    ax2 = ax1.twinx()
    ax2.plot(df["generation"], df["expected_heterozygosity"],
             color="green", label="Expected heterozygosity")
    ax2.set_ylabel("Expected heterozygosity")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
    bbox_to_anchor=(1.12, 0.5))



    plt.savefig(project_root / "figures" / exp_name /fig_name,
                dpi=300, bbox_inches="tight")

def pop_size_only_fig(df, project_root, fig_name, exp_name):
    fig, ax1 = plt.subplots()

    # Left axis
    ax1.plot(df["generation"], df["n"], color="orange", label="population size (N)")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population size")

    ax1.legend( loc="center left", bbox_to_anchor=(1.12, 0.5))


    plt.savefig(project_root / "figures" / exp_name /fig_name,
                dpi=300, bbox_inches="tight")




def expected_heterozygosity_fig(df, project_root, fig_name, exp_name):
    fig, ax1 = plt.subplots()

    ax1.plot(df["generation"], df["expected_heterozygosity"], color="green",
             label="Expected heterozygosity (queens)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Expected heterozygosity")

    # Right axis
    ax2 = ax1.twinx()
    ax2.plot(df["generation"], df["nf"], color="red",
             label="Number of colonies (breeding queens)", alpha=0.3)
    ax2.set_ylabel("Number of colonies")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               bbox_to_anchor=(1.12, 0.5))

    plt.savefig(project_root / "figures" / exp_name / fig_name,
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def colony_size_distribution_fig(df, project_root, fig_name, exp_name):

    colony_sizes = df["cs"]

    mean_colony_size = []
    sd_colony_size = []

    for gen in colony_sizes:
        mean_colony_size.append(np.mean(gen))
        sd_colony_size.append(np.std(gen))

    fig, ax1 = plt.subplots()

    # Left axis
    ax1.plot(df["generation"], mean_colony_size, color="red",
                label="Mean colony size")
    ax1.fill_between(
        df["generation"],
        np.array(mean_colony_size) - np.array(sd_colony_size),
        np.array(mean_colony_size) + np.array(sd_colony_size),
        color="red",
        alpha=0.2,
        label="± sd"
    )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Individuals (males and worker females)")

    # Right axis
    ax2 = ax1.twinx()
    ax2.plot(df["generation"], df["nf"], color="blue",
             label="Number of colonies (breeding queens)")
    ax2.set_ylabel("Number of colonies")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               bbox_to_anchor=(1.12, 0.5))

    plt.savefig(project_root / "figures" / exp_name / fig_name,
                dpi=300, bbox_inches="tight")


def queen_production_fig(df, project_root, fig_name, exp_name):
    new_queens = df["new_queens"]
    raw_queens = df["raw_queens"]

    mean_new_queens = []
    sd_new_queens = []
    mean_raw_queens = []
    sd_raw_queens = []

    for gen in new_queens:
        mean_new_queens.append(np.mean(gen))
        sd_new_queens.append(np.std(gen))

    for gen in raw_queens:
        mean_raw_queens.append(np.mean(gen))
        sd_raw_queens.append(np.std(gen))

    fig, ax1 = plt.subplots()

    ax1.plot(df["generation"], mean_raw_queens, color="blue",
             label="Queens produced colony")
    ax1.fill_between(
        df["generation"],
        np.array(mean_raw_queens) - np.array(sd_raw_queens),
        np.array(mean_raw_queens) + np.array(sd_raw_queens),
        color="blue",
        alpha=0.2,
        label="Raw ± SD"
    )

    ax1.plot(df["generation"], mean_new_queens, color="orange",
             label="Queens that survive and mate in the next generation")
    ax1.fill_between(
        df["generation"],
        np.array(mean_new_queens) - np.array(sd_new_queens),
        np.array(mean_new_queens) + np.array(sd_new_queens),
        color="orange",
        alpha=0.2,
        label="Surviving ± SD"
    )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Queens per colony")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="center left",
               bbox_to_anchor=(1.12, 0.5))

    plt.savefig(project_root / "figures" / exp_name / fig_name,
                dpi=300, bbox_inches="tight")
