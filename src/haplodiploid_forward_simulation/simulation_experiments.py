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
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def parse_to_dict(data):
    lines = []
    line = ""
    dict = {"generation" : [],
            "fixedMutations": [],
            "NucleotideHeterozygosity" : [],
            "nf" : [],
            "nm" : [],
            "ne" : [],
            "n" : []}

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

    for line in data_only:
        line = line.split()
        dict["generation"].append(int(line[0]))
        dict["fixedMutations"].append(int(line[1]))
        dict["NucleotideHeterozygosity"].append(float(line[2]))
        dict["nf"].append(float(line[3]))
        dict["nm"].append(float(line[4]))
        dict["ne"].append(float(line[5]))
        dict["n"].append(float(line[6]))

    return dict

def numpy_to_eidos(array):
    list = array.tolist()
    eidos_vector = "c("
    for element in list:
            eidos_vector = eidos_vector + str(round(element, 5)) + ", "
    eidos_vector = eidos_vector[:-2] + ")"
    return eidos_vector

def run_simulation(slim_file, slim_executable, simulation_len, k_traj, c_traj):
    command = [
        str(slim_executable),
        "-d", f"simulation_length={simulation_len}",
        "-d", f"K_traj={k_traj}",
        "-d", f"C_traj={c_traj}",
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


def main() -> None:
    # Input parameters
    file_name = argv[1]
    simulation_len = int(argv[2])

    burn_in_time = 1000
    burn_in_k = 7000
    burn_in_c = 1

    # # Burn in
    # k_burn_in = np.tile(burn_in_k, burn_in_time)
    # c_burn_in = np.tile(burn_in_c, burn_in_time)
    #
    # # Trajectory creation
    # k_sim = np.linspace(burn_in_k, 1000, simulation_len - burn_in_time)
    # c_sim = np.linspace(1, 1.5, simulation_len - burn_in_time)
    #
    # # Combine trajectories
    # k_traj = np.concatenate((k_burn_in, k_sim))
    # c_traj = np.concatenate((c_burn_in, c_sim))


    ##### No BURN IN
    k_traj = np.tile(burn_in_k, simulation_len)
    c_traj = np.tile(burn_in_c, simulation_len)


    k_traj_vector = numpy_to_eidos(k_traj)
    c_traj_vector = numpy_to_eidos(c_traj)

    project_root = Path(__file__).resolve().parents[2]

    slim_file = project_root / "slim" / file_name

    # local
    #slim_executable = Path.home() / "alt_software_versions/SLiM.V3.7/SLiM/build/slim"

    # on server
    slim_executable = Path.home() / "software/slim3.7/bin/slim"

    data = run_simulation(slim_file, slim_executable, simulation_len, k_traj_vector, c_traj_vector)
    simulation_dict = (parse_to_dict(data))

    pd_df = pd.DataFrame.from_dict(simulation_dict)
    ## add K and C trajector
    pd_df["K"] = k_traj
    pd_df["C"] = c_traj

    ## Ne plot
    fig, ax1 = plt.subplots()

    # Left axis
    ax1.plot(pd_df["generation"], pd_df["ne"], color="blue", label="Ne")
    ax1.plot(pd_df["generation"], pd_df["nf"], color="red", label="Breeding female")
    ax1.plot(pd_df["generation"], pd_df["nm"], color="green", label="Breeding male ")
    ax1.plot(pd_df["generation"], pd_df["n"], color="orange", label="Ntotal")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population size")
    ax1.xaxis.set_major_locator(MultipleLocator(10))

    # # Right axis
    # ax2 = ax1.twinx()
    #
    # ax2.plot(pd_df["generation"], pd_df["C"],
    #          color="purple", label="K")
    #
    # ax2.set_ylabel("Interspecific competition")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()

    #ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.legend()

    plt.savefig(project_root / "figures" / "K3000_C1_V1.4.png",
                dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()







