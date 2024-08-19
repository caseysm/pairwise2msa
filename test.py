import csv
import cProfile
import pstats
import io
import os
from collections import defaultdict
from pairwise2msa import main as pairwise2msa_main

def run_test(csv_file):
    # Dictionary to store cumulative time for each function
    function_times = defaultdict(float)

    # Read the CSV file
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) != 1:
                print(f"Skipping invalid row: {row}")
                continue

            base_dir = row[0]
            tree_file = os.path.join(base_dir, "rooted_upgma_tree.tree")
            fasta_folder = os.path.join(base_dir, "pairwise_fastas")
            pdb_dir = os.path.join(base_dir, "pdbs")

            print(f"\nRunning test for directory: {base_dir}")
            print(f"Tree file: {tree_file}")
            print(f"FASTA folder: {fasta_folder}")
            print(f"PDB directory: {pdb_dir}")

            # Check if all required paths exist
            if not all(os.path.exists(path) for path in [tree_file, fasta_folder, pdb_dir]):
                print(f"Skipping this directory as one or more required paths do not exist.")
                continue

            # Set up the profiler
            pr = cProfile.Profile()
            pr.enable()

            # Run the main function
            try:
                alignment, score = pairwise2msa_main(tree_file, fasta_folder, pdb_dir)
                print("\nAlignment completed successfully.")
                print(f"Final alignment length: {alignment.get_alignment_length()}")
                print(f"Number of sequences: {len(alignment)}")
                print("Scores:", score)
            except Exception as e:
                print(f"Error occurred: {str(e)}")

            pr.disable()

            # Get the stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()

            # Process the stats
            for line in s.getvalue().split('\n'):
                if line.strip() and not line.startswith('ncalls'):
                    fields = line.split()
                    if len(fields) == 6:
                        func_name = ' '.join(fields[5:])
                        time = float(fields[3])
                        function_times[func_name] += time

    # Sort functions by cumulative time
    sorted_functions = sorted(function_times.items(), key=lambda x: x[1], reverse=True)

    # Print the top 20 most time-consuming unique processes
    print("\nTop 20 most time-consuming unique processes across all runs:")
    for i, (func, time) in enumerate(sorted_functions[:20], 1):
        print(f"{i}. {func}: {time:.4f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tests for pairwise2msa script")
    parser.add_argument("csv_file", help="Path to the CSV file containing test directories")
    args = parser.parse_args()

    run_test(args.csv_file)