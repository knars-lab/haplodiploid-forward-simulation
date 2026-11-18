import numpy as np


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
            "new_queens" : [],
            "raw_queens": [],
                 }

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
        raw_queens_temp = []
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

        raw_queens = str(line[9][1:-1])
        for x in raw_queens.split(","):
            x = x.strip("'")
            if x:
                raw_queens_temp.append(int(x))
        data_dict["raw_queens"].append(raw_queens_temp)

    return data_dict

def numpy_to_eidos(array):
    list = array.tolist()
    eidos_vector = "c("
    for element in list:
            eidos_vector = eidos_vector + str(round(element, 5)) + ", "
    eidos_vector = eidos_vector[:-2] + ")"
    return eidos_vector


def get_traj_no_burn_in(simulation_len, carrying_capacity, inter_comp = 1):
    k_traj = np.tile(carrying_capacity, simulation_len)
    c_traj = np.tile(inter_comp, simulation_len)

    k_traj_vector = numpy_to_eidos(k_traj)
    c_traj_vector = numpy_to_eidos(c_traj)

    return k_traj_vector, c_traj_vector


def get_traj(simulation_len,
             burn_in_time,
             burn_in_k,
             k_bottlenecks,
             k_bottleneck_min,
             c_bottlenecks = None,
             c_traj = False):
    burn_in_c = 1

    if c_traj:
        print("hello")
        # still to do: build the C trajectory from c_bottlenecks
        return None, None

    else:
        k_burn_in = np.tile(burn_in_k, burn_in_time)
        k_traj = np.array([])
        c_traj_values = np.tile(burn_in_c, simulation_len)

        # generations per bottleneck, and the point where K is lowest
        length = int((simulation_len - burn_in_time) / k_bottlenecks)
        mid_point = int(length / 2)

        for i in range(k_bottlenecks):
            traj_cur_1 = np.linspace(burn_in_k, k_bottleneck_min, mid_point)
            traj_cur_2 = traj_cur_1[::-1]
            traj_cur = np.concatenate((traj_cur_1, traj_cur_2))
            k_traj = np.concatenate((k_traj, traj_cur))

        k_traj = np.concatenate((k_burn_in, k_traj))

        # int rounds down edge case
        if len(k_traj) < simulation_len:
            k_traj = np.concatenate(
                (k_traj, np.tile(burn_in_k, simulation_len - len(k_traj))))

        k_traj_vector = numpy_to_eidos(k_traj)
        c_traj_vector = numpy_to_eidos(c_traj_values)

        return k_traj_vector, c_traj_vector, k_traj, c_traj_values


