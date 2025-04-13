import os
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Analyse SPARC galaxies with one GP per mock sample.")
    parser.add_argument('--galaxy', type=str, default="", help='Galaxy name to analyze.')
    parser.add_argument('--redo-GPR', default=False, type=bool, help='Redo GPR analysis.')
    parser.add_argument('--server', required=True, type=str, help='Server to run the analysis on.')
    parser.add_argument('--testing', default=False, type=bool, help='Test analyses on the first 5 galaxies only.')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    server = args.server
    redo_GPR = args.redo_GPR

    galaxy = args.galaxy
    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    if galaxy != "":
        if galaxy not in galaxies:
            raise ValueError(f"Galaxy {galaxy} not found in the list of galaxies.")
        else:
            galaxies = [galaxy]
            print(f"Running analysis on {galaxy}")
    galaxy_count = len(galaxies)

    if args.testing: galaxy_count = 5

    for i in range(galaxy_count):
        g = galaxies[i]
        os.system(f'addqueue -q {server} -c "12 hrs" -n 1 -m 4 "/mnt/users/koe/SPARC_fullGP.py" --galaxy {g} --redo-GPR {str(redo_GPR)}')
