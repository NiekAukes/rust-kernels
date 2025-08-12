# This python file is used to run the benchmarks
# of Rust-gpuhc, CUDA C++, and Numba Python.
# it publishes a complete report of the benchmarks
# including figures

import os
import subprocess
import sys

import matplotlib.pyplot as plt

import re
from matplotlib.ticker import StrMethodFormatter, NullFormatter

def run_Rust_benchmarks():
    """
    Run the Rust benchmarks
    """
    print("Running Rust benchmarks...")
    os.chdir("rust")
    #subprocess.run(["cargo", "run", "--release"])
    os.chdir("..")

    # retrieve the results
    # they are stored in bench_results.txt
    rust_results = {
        "B1": [],
        "B2": [],
    }
    results = ""
    with open("rust/bench_results.txt", "r") as f:
        results = f.read()

    results = results.split("\n")
    for line in results:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the benchmark name
        benchmark_name = line[0]
        # second element is the exact benchmark
        benchmark_exact = line[1]
        # third element is the exact size of N
        benchmark_size = line[2]
        # fourth element is the time
        benchmark_time = line[3]

        rust_results[benchmark_name].append((benchmark_size,float(benchmark_time)))

    return rust_results

def run_CUDA_benchmarks():
    """
    Run the CUDA benchmarks
    """
    print("Running CUDA benchmarks...")
    os.chdir("cudacpp")
    # subprocess.run(["nvcc", "-o", "b1", "bench1.cu"], check=True)
    # subprocess.run(["./b1"])
    # subprocess.run(["nvcc", "-o", "b2", "bench2.cu"])
    # subprocess.run(["./b2"])
    os.chdir("..")

    # retrieve the results
    # they are stored in bench_results.txt
    cuda_results = {
        "B1": [],
        "B2": [],
    }
    results = ""
    with open("cudacpp/bench_results1.txt", "r") as f:
        results = f.read()
    with open("cudacpp/bench_results2.txt", "r") as f:
        results += "\n" + f.read()
    results = results.split("\n")
    for line in results:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the benchmark name
        benchmark_name = line[0]
        # second element is the exact benchmark
        benchmark_exact = line[1]
        # third element is the exact size of N
        benchmark_size = line[2]
        # fourth element is the time
        benchmark_time = line[3]

        cuda_results[benchmark_name].append((benchmark_size,float(benchmark_time)))

    return cuda_results

def run_Numba_benchmarks():
    """
    Run the Numba benchmarks
    """
    print("Running Numba benchmarks...")
    os.chdir("numba")
    # activate the venv
    # subprocess.run(["source", "venv/bin/activate"], shell=True)
    # # run the benchmarks
    # subprocess.run(["python", "main.py"], check=True)
    # # deactivate the venv
    # subprocess.run(["deactivate"], shell=True)
    os.chdir("..")



    # retrieve the results
    # they are stored in bench_results.txt
    cuda_results = {
        "B1": [],
        "B2": [],
    }
    results = ""
    with open("numba/bench_results.txt", "r") as f:
        results = f.read()
    results = results.split("\n")
    for line in results:
        if line == "":
            continue
        # split the line by |
        line = line.split("|")
        # first element is the benchmark name
        benchmark_name = line[0]
        # second element is the exact benchmark
        benchmark_exact = line[1]
        # third element is the exact size of N
        benchmark_size = line[2]
        # fourth element is the time
        benchmark_time = line[3]

        cuda_results[benchmark_name].append((benchmark_size,float(benchmark_time)))

    return cuda_results



def plot_benchmark(benchmark_dict, title="Benchmark Results", label="Benchmark"):
    sizes = []
    times = []

    for benchmark in benchmark_dict:
        # benchmark is a tuple of (size, time)
        size, time = benchmark
        sizes.append(size)
        times.append(time)


    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o', label=label)

    ax=plt.gca()
    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())

    plt.xlabel("Matrix Size (N for NxN)")
    plt.ylabel("Time (ms)")
    plt.title(title)
    plt.xticks(sizes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rust_benchmarks_linear.png")
    print("Close the plot to continue...")
    plt.show()


def plot_benchmarks(all_benchmarks, title="Benchmark Results (Log Y)"):
    plt.figure(figsize=(10, 6))

    for lang, benchmark_dict in all_benchmarks.items():
        sizes = []
        times = []

        for (size, time) in benchmark_dict:
            sizes.append(size)
            times.append(time)


        plt.plot(sizes, times, marker='o', label=lang)

    # ax = plt.gca()
    # ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # ax.yaxis.set_minor_formatter(NullFormatter())

    plt.xlabel("Matrix Size (N for NxN)")
    plt.ylabel("Time (ms)")
    plt.title(title)
    plt.xticks(sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_benchmarks_log.png")
    print("Close the plot to continue...")
    plt.show()


def plot_benchmarks_baseline(baseline, all_benchmarks, title="Benchmark Results (Log Y)"):
    plt.figure(figsize=(10, 6))

    baseline_times = {}
    for benchmark in baseline:
        # benchmark is a tuple of (size, time)
        size, time = benchmark
        baseline_times[size] = time

    for lang, benchmark_dict in all_benchmarks.items():
        sizes = []
        times = []

        for (size, time) in benchmark_dict:
            sizes.append(size)
            times.append(time / baseline_times[size])


        plt.plot(sizes, times, marker='o', label=lang)

    # ax = plt.gca()
    # ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # ax.yaxis.set_minor_formatter(NullFormatter())

    plt.xlabel("Matrix Size (N for NxN)")
    plt.ylabel("Time (ms)")
    plt.title(title)
    plt.xticks(sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_benchmarks_log.png")
    print("Close the plot to continue...")
    plt.show()

if __name__ == "__main__":
    #os.chdir("benches")
    results_rust = run_Rust_benchmarks()
    results_cuda = run_CUDA_benchmarks()
    results_numba = run_Numba_benchmarks()
    # plot_benchmarks({
    #     "Rust": results_rust["B1"],
    #     "CUDA": results_cuda["B1"],
    #     "Numba": results_numba["B1"],
    # }, title="Benchmark Results (Log Y)")
    plot_benchmarks_baseline(results_cuda["B1"], {
        "Rust": results_rust["B1"],
        "CUDA": results_cuda["B1"],
        "Numba": results_numba["B1"],
    }, title="Benchmark Results (Log Y)")
    plot_benchmarks_baseline(results_cuda["B2"], {
        "Rust": results_rust["B2"],
        "CUDA": results_cuda["B2"],
        "Numba": results_numba["B2"],
    }, title="Benchmark Results (Log Y)")
    #os.chdir("..")