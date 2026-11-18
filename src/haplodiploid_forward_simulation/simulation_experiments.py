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
from figs import *
from utils import *

from sys import argv
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import resource
import time
from sklearn.model_selection import ParameterGrid




def run_simulation(slim_file, slim_executable, simulation_len, k_traj, c_traj,
                   start_population_size, starting_virgin_queens, max_queens_per_colony,
                   max_colony_size, sex_ratio, intra_specific_comp, p_colony_size_scaling,
                   p_population_size_scaling, p_new_queen_scaling, p_queen_success_scaling,
                   mutation_rate, c_intra_threshold, recombination_rate, sd_control_queen, sd_control_colony):
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
        "-d", f"sd_control_queen={sd_control_queen}",
        "-d", f"sd_control_colony={sd_control_colony}",
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

                # skip the first generation
                df = df[df["generation"] > 30].reset_index(drop=True)

                colony_size_distribution_fig(df, project_root, f"{run_name}_colony_size_distribution", exp_name)
                queen_production_fig(df, project_root,
                                             f"{run_name}_queen_production",
                                             exp_name)
                pop_size_fig(df, project_root, f"{run_name}_population_size", exp_name)
                pop_size_only_fig(df, project_root, f"{run_name}_population_size_only", exp_name)
                expected_heterozygosity_fig(df, project_root,
                                            f"{run_name}_heterozygosity",
                                            exp_name)
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
        "max_queens_per_colony": 100,
        "max_colony_size": 200,
        "sex_ratio": 0.3,
        "intra_specific_comp": "F",
        "p_colony_size_scaling": 0.8,
        "c_intra_threshold": 0.8,
        "p_queen_success_scaling": 0.8,
        "p_population_size_scaling": 1.0,
        "p_new_queen_scaling": 0.5,
        "mutation_rate": 3.6e-9,
        "recombination_rate": 9e-8,
        "sd_control_queen": 50,
        "sd_control_colony": 50,
    }

    grid = ParameterGrid({
        "p_colony_size_scaling": [0.4, 0.75],
        "p_population_size_scaling": [0.7, 1.0],
    })

    run_experiments(argv[4], project_root, slim_file, slim_executable, grid, defaults,
                    simulation_len, carrying_capacity = 30000, burn_in=False)


    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f"Peak RAM of SLiM: {usage.ru_maxrss / 1024:.0f} MB")
if __name__ == "__main__":
    main()







