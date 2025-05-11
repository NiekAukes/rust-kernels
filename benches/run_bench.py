# This python file is used to run the benchmarks
# of Rust-gpuhc, CUDA C++, and Numba Python.
# it publishes a complete report of the benchmarks
# including figures

import os
import subprocess
import sys


def run_Rust_benchmarks():
    """
    Run the Rust benchmarks
    """
    print("Running Rust benchmarks...")
    os.chdir("rust")
    subprocess.run(["cargo", "run", "--release"])
    os.chdir("..")

    # retrieve the results