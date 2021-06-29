import subprocess
from os import listdir, mkdir
from os.path import isfile, join, dirname, isdir
from datetime import datetime


def get_abs_path(file_or_dir: str) -> str:
    return join(dirname(__file__), file_or_dir)


def run_parallel(threads=4):
    # Prepare input file
    input_path = get_abs_path('./data')
    input_files = sorted([join(input_path, f) for f in listdir(input_path)
                          if isfile(join(input_path, f)) and f.endswith(".txt")])

    # Prepare output file
    output_path = get_abs_path('./out')
    if not isdir(output_path):
        mkdir(output_path)
    output_file = join(output_path, f'biip_parallel_output_{threads}.txt')

    # Write a timestamp of current date to output file
    with open(get_abs_path(output_file), 'a') as out:
        print(f'{"-" * 40} {datetime.now()} {"-" * 40}', file=out)

    # Execute BIIP_Parallel for input files
    for input_file in input_files:
        make = subprocess.Popen(["make"])
        make.wait()
        proc = subprocess.Popen(["./biip", str(threads), input_file, output_file])
        proc.wait()


def run_sequential():
    # Prepare input file
    input_path = get_abs_path('./data')
    input_files = sorted([join(input_path, f) for f in listdir(input_path)
                          if isfile(join(input_path, f)) and f.endswith(".txt")])

    # Prepare output file
    output_path = get_abs_path('./out')
    if not isdir(output_path):
        mkdir(output_path)
    output_file = join(output_path, f'biip_output.txt')

    # Write a timestamp of current date to output file
    with open(get_abs_path(output_file), 'a') as out:
        print(f'{"-" * 40} {datetime.now()} {"-" * 40}', file=out)

    # Execute BIIP_Parallel for input files
    for input_file in input_files:
        make = subprocess.Popen(["make", "-C", get_abs_path('./biip_original')])
        make.wait()
        proc = subprocess.Popen(["./biip_original/biip", input_file, output_file])
        proc.wait()


if __name__ == '__main__':
    run_parallel(threads=8)
    # run_sequential()
