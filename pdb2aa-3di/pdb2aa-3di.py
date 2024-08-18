import os
import re
import time
import argparse
import numpy as np
import subprocess
import shutil

# Update this line to use Conda-installed Foldseek
FOLDSEEK_CMD = "foldseek"


def get_struc_seq(path, chains=None, process_id=0, plddt_mask=False, plddt_threshold=70.,
                  foldseek_verbose=False):
    assert shutil.which(
        FOLDSEEK_CMD) is not None, f"Foldseek not found in PATH. Make sure it's installed with Conda and the environment is activated."
    assert os.path.exists(path), f"PDB file not found: {path}"

    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    cmd = [FOLDSEEK_CMD, "structureto3didescriptor", "--threads", "1", "--chain-name-mode", "1", path, tmp_save_path]

    if not foldseek_verbose:
        cmd.insert(2, "-v")
        cmd.insert(3, "0")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Foldseek: {e}")
        print(f"Foldseek stdout: {e.stdout}")
        print(f"Foldseek stderr: {e.stderr}")
        raise

    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]

            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"

                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)

            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)

    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


def extract_plddt(pdb_path):
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(' +', ' ', line).strip()
            splits = line.split(" ")

            if splits[0] == "ATOM":
                if len(splits[4]) == 1:
                    pos = int(splits[5])
                else:
                    pos = int(splits[4][1:])

                plddt = float(splits[-2])

                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)

    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts


def read_fasta(file_path):
    sequences = {}
    current_id = None
    current_seq = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id:
        sequences[current_id] = ''.join(current_seq)

    return sequences


def convert_fasta_to_3di(fasta_dir, pdb_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith('.fasta') or fasta_file.endswith('.fa'):
            fasta_path = os.path.join(fasta_dir, fasta_file)
            sequences = read_fasta(fasta_path)

            if len(sequences) not in [1, 2]:
                print(f"Skipping {fasta_file}: Expected 1 or 2 sequences, found {len(sequences)}")
                continue

            threedi_sequences = {}
            for seq_id, seq in sequences.items():
                pdb_file = os.path.join(pdb_dir, f"{seq_id}.pdb")
                if not os.path.exists(pdb_file):
                    print(f"Skipping {fasta_file}: PDB file not found for {seq_id}")
                    break

                seq_dict = get_struc_seq(pdb_file)
                if not seq_dict:
                    print(f"Skipping {fasta_file}: Could not extract 3Di sequence for {seq_id}")
                    break

                # Assume we're working with the first chain
                chain = list(seq_dict.keys())[0]
                _, threedi_seq, _ = seq_dict[chain]

                # Align 3Di sequence to match the FASTA alignment
                aligned_threedi = ''.join([threedi_seq[seq.index(aa)] if aa != '-' else '-' for aa in seq])
                threedi_sequences[seq_id] = aligned_threedi

            if len(threedi_sequences) == len(sequences):
                output_file = os.path.join(output_dir, f"{os.path.splitext(fasta_file)[0]}_3di.fasta")
                with open(output_file, 'w') as f:
                    for seq_id, threedi_seq in threedi_sequences.items():
                        f.write(f">{seq_id}\n{threedi_seq}\n")
                print(f"Converted {fasta_file} to 3Di alignment")
            else:
                print(f"Skipping {fasta_file}: Could not process all sequences")


def main(fasta_dir=None, pdb_dir=None, output_dir=None):
    if not all([fasta_dir, pdb_dir, output_dir]):
        print("Error: You must specify FASTA directory, PDB directory, and output directory.")
        return

    if not os.path.exists(fasta_dir):
        print(f"Error: FASTA directory not found: {fasta_dir}")
        return

    if not os.path.exists(pdb_dir):
        print(f"Error: PDB directory not found: {pdb_dir}")
        return

    convert_fasta_to_3di(fasta_dir, pdb_dir, output_dir)


# Example usage when running from an IDE
if __name__ == "__main__":
    # Replace these with your actual directory paths
    example_fasta_dir = "/home/casey/Desktop/lab_projects/test/pdb2pairwise/usalign_fNS/pairwise_fastas"
    example_pdb_dir = "/home/casey/Desktop/lab_projects/test/pdbs"
    example_output_dir = "/home/casey/Desktop/lab_projects/test/pdb2pairwise/usalign_fNS/pairwise_fastas_3di"

    main(fasta_dir=example_fasta_dir, pdb_dir=example_pdb_dir, output_dir=example_output_dir)