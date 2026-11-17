#!/usr/bin/env python3
"""
Run a SLiM forward simulation and generate experimental plots.

Usage:
    uv run python simulation_experiments.py <slim_file> <simulation_length>

Arguments:
    slim_file:
        Name of the SLiM input file located in the project's slim/ directory.

    simulation_length:
        Number of generations to simulate.

Example:
    uv run python simulation_experiments.py bumble_bee_forward_simulation.slim 1000

Output:
    Saves a figures in the project's figures/ directory.
"""

from sys import argv
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import resource
import time
from sklearn.model_selection import ParameterGrid


def parse_to_dict(data):
    lines = []
    line = ""
    data_dict = {"generation" : [],
            "fixedMutations": [],
            "expected_heterozygosity" : [],
            "nf" : [],
            "nm" : [],
            "ne" : [],
            "n" : [],
            "cs" : [],
            "new_queens" : []}

    for element in data:
        if element != "\n":
            line += element
        else:
            lines.append(line)
            line = ""

    for index, line in enumerate(lines):
        if line.startswith("Generation FixedMutations"):
            data_only = lines[index + 1:]
            break

    for i, line in enumerate(data_only):
        colony_size_temp = []
        new_queens_temp = []
        if line.startswith("extinction at"):
            continue
        line = line.split()
        data_dict["generation"].append(int(line[0]))
        data_dict["fixedMutations"].append(int(line[1]))
        data_dict["expected_heterozygosity"].append(float(line[2]))
        data_dict["nf"].append(float(line[3]))
        data_dict["nm"].append(float(line[4]))
        data_dict["ne"].append(float(line[5]))
        data_dict["n"].append(float(line[6]))
        colony_size = str(line[7][1:-1])

        for x in colony_size.split(","):
            x = x.strip("'")
            if x:
                colony_size_temp.append(int(x))
        data_dict["cs"].append(colony_size_temp)
        new_queens = str(line[8][1:-1])
        for x in new_queens.split(","):
            x = x.strip("'")
            if x:
                new_queens_temp.append(int(x))
        data_dict["new_queens"].append(new_queens_temp)

    return data_dict

def numpy_to_eidos(array):
    list = array.tolist()
    eidos_vector = "c("
    for element in list:
            eidos_vector = eidos_vector + str(round(element, 5)) + ", "
    eidos_vector = eidos_vector[:-2] + ")"
    return eidos_vector

def run_simulation(slim_file, slim_executable, simulation_len, k_traj, c_traj,
                   start_population_size, starting_virgin_queens, max_queens_per_colony,
                   max_colony_size, sex_ratio, intra_specific_comp, p_colony_size_scaling,
                   p_population_size_scaling, p_new_queen_scaling, p_queen_success_scaling,
                   mutation_rate, c_intra_threshold, recombination_rate):
    command = [
        str(slim_executable),
        "-d", f"simulation_length={simulation_len}",
        "-d", f"K_traj={k_traj}",
        "-d", f"C_traj={c_traj}",
        "-d", f"start_population_size={start_population_size}",
        "-d", f"starting_virgin_queens={starting_virgin_queens}",
        "-d", f"max_queens_per_colony={max_queens_per_colony}",
        "-d", f"max_colony_size={max_colony_size}",
        "-d", f"sex_ratio={sex_ratio}",
        "-d", f"intra_specific_comp={intra_specific_comp}",
        "-d", f"p_colony_size_scaling={p_colony_size_scaling}",
        "-d", f"p_population_size_scaling={p_population_size_scaling}",
        "-d", f"p_new_queen_scaling={p_new_queen_scaling}",
        "-d", f"p_queen_success_scaling={p_queen_success_scaling}",
        "-d", f"mutation_rate={mutation_rate}",
        "-d", f"C_intra_threshold={c_intra_threshold}",
        "-d", f"recombination_rate={recombination_rate}",
        str(slim_file)
    ]

    output = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    if output.returncode != 0:
        print("SLiM ERROR:")
        print(output.stderr)
        raise RuntimeError("SLiM simulation failed")

    return output.stdout


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
             color="orange", label="Expected heterozygosity")
    ax2.set_ylabel("Expected heterozygosity")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
    bbox_to_anchor=(1.12, 0.5))



    plt.savefig(project_root / "figures" / exp_name /fig_name,
                dpi=300, bbox_inches="tight")




def expected_heterozygosity_fig(df, project_root, fig_name, exp_name):
    fig, ax1 = plt.subplots()

    # Left axis
    ax1.plot(df["generation"], df["expected_heterozygosity"], color="blue", label="Effective population size (Ne)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population size (log10)")
    ax1.set_yscale("log")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()

    ax1.legend(lines1 , labels1, loc="center left",
    bbox_to_anchor=(1.12, 0.5))

    plt.savefig(project_root / "figures" / exp_name / fig_name,
                dpi=300, bbox_inches="tight")


def colony_size_distribution_fig(df, project_root, fig_name, exp_name):

    colony_sizes = df["cs"]

    mean_colony_size = []
    sd_colony_size = []

    for gen in colony_sizes:
        mean_colony_size.append(np.mean(gen))
        sd_colony_size.append(np.std(gen))

    new_queens = df["new_queens"]
    mean_new_queens = []
    sd_new_queens = []

    for gen in new_queens:
        mean_new_queens.append(np.mean(gen))
        sd_new_queens.append(np.std(gen))

    fig, ax1 = plt.subplots()

    # Left axis
    ax1.plot(df["generation"], mean_colony_size, color="red",
             label="Mean colony size")
    ax1.plot(df["generation"], mean_new_queens, color="blue",
             label="Queen production")

    ax1.fill_between(
        df["generation"],
        np.array(mean_colony_size) - np.array(sd_colony_size),
        np.array(mean_colony_size) + np.array(sd_colony_size),
        color="red",
        alpha=0.2,
        label="± sd"
    )

    ax1.fill_between(
        df["generation"],
        np.array(mean_new_queens) - np.array(sd_new_queens),
        np.array(mean_new_queens) + np.array(sd_new_queens),
        color="blue",
        alpha=0.2,
        label="± SD"
    )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Individuals")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()

    ax1.legend(lines1, labels1, loc="center left",
               bbox_to_anchor=(1.12, 0.5))

    plt.savefig(project_root / "figures" / exp_name / fig_name,
                dpi=300, bbox_inches="tight")


def get_traj_no_burn_in(simulation_len, carrying_capacity, inter_comp = 1):
    k_traj = np.tile(carrying_capacity, simulation_len)
    c_traj = np.tile(inter_comp, simulation_len)

    k_traj_vector = numpy_to_eidos(k_traj)
    c_traj_vector = numpy_to_eidos(c_traj)

    return k_traj_vector, c_traj_vector


def get_traj(simulation_len):
    burn_in_time = 1000
    burn_in_k = 50000
    burn_in_c = 1

    # Comment out for pinned k and c
    # create burn in and concatenate with trajectories
    # k_burn_in = np.tile(burn_in_k, burn_in_time)
    # c_burn_in = np.tile(burn_in_c, burn_in_time)
    # k_sim = np.linspace(burn_in_k, 1000, simulation_len - burn_in_time)
    # c_sim = np.linspace(1, 1.5, simulation_len - burn_in_time)
    # k_traj = np.concatenate((k_burn_in, k_sim))
    # c_traj = np.concatenate((c_burn_in, c_sim))

    # comment out if trajectories for C and K are needed
    k_traj = np.tile(burn_in_k, simulation_len)
    c_traj = np.tile(burn_in_c, simulation_len)

    # trajectories to eidos vector
    k_traj_vector = numpy_to_eidos(k_traj)
    c_traj_vector = numpy_to_eidos(c_traj)

    return k_traj_vector, c_traj_vector



def run_experiments(exp_name,  project_root, slim_file, slim_executable, grid,
                    defaults, simulation_len, carrying_capacity,
                    burn_in = False):

    paramset_count = len(grid)

    experiment_dir = project_root / "figures" / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / f"{exp_name}_meta_data", "w") as file:
        file.write(f"Meta data {exp_name}\n")

        if burn_in:
            print("burn_in true")
        else:
            for i, param_set in enumerate(grid):

                params = {**defaults, **param_set}
                run_name = f"{exp_name}_{i+1}"
                file.write(f"{run_name} : {carrying_capacity} | {params}\n")

                start = time.perf_counter()

                k_traj, c_traj = get_traj_no_burn_in(simulation_len,
                                                     carrying_capacity)
                data = run_simulation(slim_file, slim_executable, simulation_len, k_traj, c_traj, **params)

                stop =  time.perf_counter()
                duration = stop - start
                minutes, seconds = divmod(duration, 60)
                file.write(f"{run_name} : run time: {int(minutes)} min {seconds:.2f} sec\n")
                simulation_dict = (parse_to_dict(data))

                df = pd.DataFrame.from_dict(simulation_dict)
                ## add K and C trajector
                df["K"] = k_traj[:len(df)]
                df["C"] = c_traj[:len(df)]
                colony_size_distribution_fig(df, project_root, f"{run_name}_colony_size_distribution", exp_name)
                pop_size_fig(df, project_root, f"{run_name}_population_size", exp_name)
                print(f"param_set {i+1} out of {paramset_count}")


def main() -> None:
    # =======================================
    # Input parameters and path definitions
    # =======================================

    file_name = argv[1]
    simulation_len = int(argv[2])
    project_root = Path(__file__).resolve().parents[2]
    slim_file = project_root / "slim" / file_name


    if argv[3] == "local":
        slim_executable = Path.home() / "alt_software_versions/SLiM.V3.7/SLiM/build/slim"
    elif argv[3] == "server":
        slim_executable = Path.home() / "software/slim3.7/bin/slim"
    else:
        print("specify server or local")
        exit()

    # ========================================
    # burn in settings and trajectory creation
    # ========================================

    defaults = {
        "start_population_size": 5000,
        "starting_virgin_queens": 300,
        "max_queens_per_colony": 200,
        "max_colony_size": 400,
        "sex_ratio": 0.3,
        "intra_specific_comp": "F",
        "p_colony_size_scaling": 0.8,
        "c_intra_threshold": 0.8,
        "p_queen_success_scaling": 0.8,
        "p_population_size_scaling": 1.0,
        "p_new_queen_scaling": 0.5,
        "mutation_rate": 3.6e-9,
        "recombination_rate": 9e-8,
    }

    grid = ParameterGrid({
        # "intra_specific_comp": ["F", "T"],
        "p_colony_size_scaling": [0.8],
        "max_colony_size": [200],
    })

    run_experiments(argv[4], project_root, slim_file, slim_executable, grid, defaults,
                    simulation_len, carrying_capacity = 30000, burn_in=False)

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f"Peak RAM of SLiM: {usage.ru_maxrss / 1024:.0f} MB")
if __name__ == "__main__":
    main()







