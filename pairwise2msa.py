import os
import sys
import re
import time
import tempfile
import subprocess
import shutil
import math
import csv
import logging
from pathlib import Path
from typing import Optional, List, Dict
from functools import lru_cache
from itertools import combinations


import numpy as np
import pandas as pd

from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.PDB import PDBParser as BioPDBParser

from ete3 import Tree



"""
************************************************************************************************************************
********************************************* TREE LOADING AND PROCESSSING *********************************************
************************************************************************************************************************
"""


def load_tree(tree_file):
    """
    Load a tree from a Newick file.

    Args:
        tree_file (str): Path to the Newick tree file.

    Returns:
        Tree: Loaded tree object.
    """
    try:
        return Tree(tree_file)
    except Exception as e:
        print(f"Error loading tree: {e}")
        return None


def label_nodes_by_leaf_descendants(node):
    """
    Recursively label nodes by concatenating the names of their leaf descendants.

    Args:
        node (Tree): Current node in the tree.

    Returns:
        set: Set of leaf names under this node.
    """
    if node.is_leaf():
        return {node.name}

    leaf_names = set()
    for child in node.children:
        leaf_names.update(label_nodes_by_leaf_descendants(child))

    node.name = "_".join(sorted(leaf_names))
    return leaf_names


def traverse_and_print(node, level=0):
    """
    Recursively traverse the tree and print node names with indentation.

    Args:
        node (Tree): Current node in the tree.
        level (int): Current depth level for indentation.
    """
    indent = "  " * level
    print(f"{indent}{node.name}")

    for child in node.children:
        traverse_and_print(child, level + 1)


def process_tree(tree_file):
    """
    Load a tree, label its nodes by leaf descendants, and print the structure.

    Args:
        tree_file (str): Path to the Newick tree file.

    Returns:
        Tree: Labeled tree object.
    """
    # Load the tree
    tree = load_tree(tree_file)

    if tree is None:
        return None

    # Label nodes by leaf descendants
    label_nodes_by_leaf_descendants(tree)

    print("Tree structure after labeling:")
    traverse_and_print(tree)

    return tree


def classify_nodes(tree):
    """
    Classify all nodes in the tree and return a dictionary of classifications.

    Args:
        tree (Tree): The root node of the tree to classify.

    Returns:
        dict: A dictionary with node names as keys and their classifications as values.
    """
    classifications = {}

    def classify_node(node):
        if node.is_leaf():
            return "leaf"

        leaf_children = [child for child in node.children if child.is_leaf()]
        node_children = [child for child in node.children if not child.is_leaf()]

        if len(node.children) == 2:
            if all(child.is_leaf() for child in node.children):
                return "leaf_binary"
            elif all(not child.is_leaf() for child in node.children):
                return "node_binary"

        if leaf_children and node_children:
            return "mixed_complex"
        elif all(child.is_leaf() for child in node.children):
            return "leaf_complex"
        else:
            return "node_complex"

    def traverse_and_classify(node):
        classification = classify_node(node)
        classifications[node.name] = classification

        for child in node.children:
            traverse_and_classify(child)

    traverse_and_classify(tree)
    return classifications


def print_ete_tree(tree):
    """
    Print the tree to the console using ETE3's ASCII art representation.

    Args:
        tree (Tree): The ETE3 Tree object to print.
    """
    print("\nTree Visualization:")
    print(tree.get_ascii(show_internal=True))


"""
************************************************************************************************************************
**************************************************** FASTA LOADING *****************************************************
************************************************************************************************************************
"""


def load_pairwise_fastas(folder_path):
    """
    Load pairwise FASTA files from a folder into a dictionary.

    Args:
        folder_path (str): Path to the folder containing pairwise FASTA files.

    Returns:
        dict: Dictionary with tuple keys of FASTA names and MultipleSeqAlignment values.
    """
    leaf_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.fasta'):
            # Extract names from the filename
            names = filename[:-6].split('_')
            if len(names) == 2:
                file_path = os.path.join(folder_path, filename)
                try:
                    # Load the alignment
                    alignment = AlignIO.read(file_path, "fasta")
                    # Store in dictionary with tuple of names as key
                    leaf_dict[tuple(names)] = alignment
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return leaf_dict


def traverse_and_match(node, leaf_dict, node_dict, level=0):
    """
    Recursively traverse the labeled tree and check for matches with the alignment dictionary.

    Args:
        node (Tree): Current node in the tree.
        leaf_dict (dict): Dictionary of pairwise alignments.
        node_dict (dict): Dictionary to store node alignments.
        level (int): Current depth level for indentation.
    """
    indent = "  " * level
    node_names = node.name.split('_')

    if len(node_names) == 2:
        if tuple(node_names) in leaf_dict or tuple(reversed(node_names)) in leaf_dict:
            print(f"{indent}{node.name}: Matched in alignment dictionary")
        else:
            print(f"{indent}{node.name}: No match found in alignment dictionary")
    else:
        print(f"{indent}{node.name}: Complex node")

    for child in node.children:
        traverse_and_match(child, leaf_dict, node_dict, level + 1)

"""
************************************************************************************************************************
******************************************************** MAFFT *********************************************************
************************************************************************************************************************
"""

def run_mafft_alignment(input_path: str, output_path: str, mafft_path: Optional[str] = None,
                        custom_matrix_path: Optional[str] = None,
                        additional_args: Optional[List[str]] = None) -> MultipleSeqAlignment:
    """
    Run MAFFT alignment on the given input file and return the aligned sequences.

    Args:
        input_path (str): Path to the input FASTA file with unaligned sequences.
        output_path (str): Path where the aligned FASTA will be saved.
        mafft_path (Optional[str]): Path to the MAFFT executable. If None, assumes 'mafft' is in PATH.
        custom_matrix_path (Optional[str]): Path to a custom substitution matrix file.
        additional_args (Optional[List[str]]): Additional command-line arguments for MAFFT.

    Returns:
        MultipleSeqAlignment: The aligned sequences.

    Raises:
        subprocess.CalledProcessError: If MAFFT fails to run.
        FileNotFoundError: If the MAFFT executable or custom matrix file is not found.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    mafft_cmd = mafft_path if mafft_path else 'mafft'

    try:
        # Construct MAFFT command
        command = [mafft_cmd, '--auto']

        if custom_matrix_path:
            if not os.path.exists(custom_matrix_path):
                raise FileNotFoundError(f"Custom matrix file not found: {custom_matrix_path}")
            command.extend(['--aamatrix', custom_matrix_path])

        if additional_args:
            command.extend(additional_args)

        command.extend([input_path])

        # Run MAFFT
        logging.info(f"Running MAFFT command: {' '.join(command)}")
        with open(output_path, 'w') as output_file:
            subprocess.run(command, check=True, stdout=output_file, stderr=subprocess.PIPE, text=True)

        # Read and return the alignment
        with open(output_path, "r") as handle:
            alignment = AlignIO.read(handle, "fasta")

        logging.info(f"MAFFT alignment completed successfully. Output written to {output_path}")
        return alignment

    except subprocess.CalledProcessError as e:
        logging.error(f"MAFFT alignment failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logging.error(f"MAFFT executable not found. Make sure it's installed and in PATH, or provide the correct path.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during MAFFT alignment: {str(e)}")
        raise

    except subprocess.CalledProcessError as e:
        logging.error(f"MAFFT alignment failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logging.error(f"MAFFT executable not found. Make sure it's installed and in PATH, or provide the correct path.")
        raise


def check_mafft_installation(mafft_path: Optional[str] = None) -> bool:
    """
    Check if MAFFT is installed and accessible.

    Args:
        mafft_path (Optional[str]): Path to the MAFFT executable. If None, assumes 'mafft' is in PATH.

    Returns:
        bool: True if MAFFT is installed and accessible, False otherwise.
    """
    mafft_cmd = mafft_path if mafft_path else 'mafft'

    try:
        result = subprocess.run([mafft_cmd, '--version'], capture_output=True, text=True)
        logging.info(f"MAFFT version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logging.warning("MAFFT is not installed or not in PATH")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking MAFFT installation: {e}")
        return False


def combine_alignments_mafft(alignments: List[MultipleSeqAlignment], aa_to_3di_map: Dict, _3di_matrix_file: Path) -> MultipleSeqAlignment:
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_input, \
            tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_output:
        # Convert AA sequences to 3Di and write to the input file
        for alignment in alignments:
            for record in alignment:
                _3di_seq = aa_to_3di(str(record.seq), record.id, aa_to_3di_map)
                temp_input.write(f">{record.id}\n{_3di_seq}\n")
        temp_input.flush()

        # Run MAFFT alignment with 3Di custom matrix
        additional_args = ['--quiet']
        result_alignment = run_mafft_alignment(temp_input.name, temp_output.name, custom_matrix_path=str(_3di_matrix_file), additional_args=additional_args)

        # Convert 3Di alignment back to AA
        aa_alignment = MultipleSeqAlignment([])
        for record in result_alignment:
            aa_seq = _3di_to_aa(str(record.seq), record.id, aa_to_3di_map)
            aa_alignment.append(SeqRecord(Seq(aa_seq), id=record.id, description=""))

        # Clean up temporary files
        os.unlink(temp_input.name)
        os.unlink(temp_output.name)

        return aa_alignment

def final_mafft_alignment(alignment: MultipleSeqAlignment, aa_to_3di_map: Dict, _3di_matrix_file: str) -> MultipleSeqAlignment:
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_input, \
            tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_output:
        # Convert AA sequences to 3Di and write to the input file
        for record in alignment:
            _3di_seq = aa_to_3di(str(record.seq), record.id, aa_to_3di_map)
            temp_input.write(f">{record.id}\n{_3di_seq}\n")
        temp_input.flush()

        additional_args = ['--maxiterate', '1000', '--quiet']
        result_alignment = run_mafft_alignment(temp_input.name, temp_output.name, custom_matrix_path=_3di_matrix_file, additional_args=additional_args)

        # Convert 3Di alignment back to AA
        aa_alignment = MultipleSeqAlignment([])
        for record in result_alignment:
            aa_seq = _3di_to_aa(str(record.seq), record.id, aa_to_3di_map)
            aa_alignment.append(SeqRecord(Seq(aa_seq), id=record.id, description=""))

        # Clean up temporary files
        os.unlink(temp_input.name)
        os.unlink(temp_output.name)

        return aa_alignment


"""
************************************************************************************************************************
************************************************* AA TO 3Di Converter **************************************************
************************************************************************************************************************
"""


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


def build_aa_to_3di_mapping(pdb_dir):
    aa_to_3di_map = {}
    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            seq_dict = get_struc_seq(pdb_path)
            for chain, (aa_seq, _3di_seq, _) in seq_dict.items():
                seq_id = os.path.splitext(pdb_file)[0]
                aa_to_3di_map[seq_id] = {
                    'aa_seq': aa_seq,
                    '3di_seq': _3di_seq
                }
    return aa_to_3di_map


def aa_to_3di(aa_seq, seq_id, aa_to_3di_map_frozen):
    aa_to_3di_map = frozenset_to_dict(aa_to_3di_map_frozen)
    if seq_id not in aa_to_3di_map:
        raise ValueError(f"Sequence ID {seq_id} not found in AA to 3Di mapping")
    aa_full = aa_to_3di_map[seq_id]['aa_seq']
    _3di_full = aa_to_3di_map[seq_id]['3di_seq']
    _3di_seq = ''
    aa_index = 0
    for aa in aa_seq:
        if aa == '-':
            _3di_seq += '-'
        else:
            if aa_index < len(aa_full):
                _3di_seq += _3di_full[aa_index]
                aa_index += 1
            else:
                # If we've run out of 3Di sequence, add 'X'
                _3di_seq += 'X'
                print(
                    f"Warning: AA sequence for {seq_id} is longer than the 3Di sequence. Using 'X' for position {aa_index + 1}.")

    # If we haven't used all of the 3Di sequence, print a warning
    if aa_index < len(_3di_full):
        print(
            f"Warning: Not all of the 3Di sequence for {seq_id} was used. {len(_3di_full) - aa_index} 3Di characters remaining.")

    return _3di_seq


def _3di_to_aa(_3di_seq, seq_id, aa_to_3di_map_frozen):
    aa_to_3di_map = frozenset_to_dict(aa_to_3di_map_frozen)
    if seq_id not in aa_to_3di_map:
        raise ValueError(f"Sequence ID {seq_id} not found in AA to 3Di mapping")
    aa_full = aa_to_3di_map[seq_id]['aa_seq']
    _3di_full = aa_to_3di_map[seq_id]['3di_seq']
    aa_seq = ''
    _3di_index = 0
    for _3di in _3di_seq:
        if _3di == '-':
            aa_seq += '-'
        else:
            if _3di_index < len(_3di_full):
                aa_seq += aa_full[_3di_index]
                _3di_index += 1
            else:
                # If we've run out of AA sequence, add 'X'
                aa_seq += 'X'
                print(
                    f"Warning: 3Di sequence for {seq_id} is longer than the AA sequence. Using 'X' for position {_3di_index + 1}.")

    # If we haven't used all of the AA sequence, print a warning
    if _3di_index < len(aa_full):
        print(
            f"Warning: Not all of the AA sequence for {seq_id} was used. {len(aa_full) - _3di_index} AA characters remaining.")

    return aa_seq

"""
************************************************************************************************************************
******************************************************* SCORING ********************************************************
************************************************************************************************************************
"""

# Constants
AA_MAPPING = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}

# Set up the correct path for the matrix directory
SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "scoring_matrix"

# Ensure MATRIX_DIR exists
if not MATRIX_DIR.exists():
    raise FileNotFoundError(f"Matrix directory not found: {MATRIX_DIR}")

# Update matrix_file to use MATRIX_DIR
matrix_file = MATRIX_DIR / "BLOSUM62.txt"

# Check if matrix_file exists
if not matrix_file.exists():
    raise FileNotFoundError(f"Matrix file not found: {matrix_file}")



class CustomPDBParser:
    @staticmethod
    def parse_pdb_sequence_and_coordinates(pdb_file):
        parser = BioPDBParser(QUIET=True)
        try:
            structure = parser.get_structure('protein', pdb_file)
        except Exception as e:
            raise ValueError(f"Error parsing PDB file {pdb_file}: {str(e)}")

        sequence = ""
        coords = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        try:
                            one_letter = AA_MAPPING.get(residue.resname, 'X')
                            sequence += one_letter
                            coords[len(sequence)] = {
                                'ca': residue['CA'].get_coord(),
                                'residue': one_letter
                            }
                        except KeyError:
                            print(f"Warning: Unknown residue {residue.resname} in {pdb_file}. Using 'X'.")
                break  # Only process the first chain
            break  # Only process the first model
        return sequence, coords


class FASTAParser:
    @staticmethod
    def parse_fasta_alignment(fasta_file):
        try:
            alignment = AlignIO.read(fasta_file, "fasta")
        except Exception as e:
            raise ValueError(f"Error reading FASTA file: {str(e)}")

        return {record.id: str(record.seq) for record in alignment}


class AlignmentEvaluator:
    @staticmethod
    def load_csv_matrix(matrix_file):
        df = pd.read_csv(matrix_file, index_col=0)
        if not df.index.equals(df.columns):
            raise ValueError("Matrix is not symmetric: row and column labels do not match")
        matrix_dict = {(aa1, aa2): df.loc[aa1, aa2]
                       for aa1 in df.index
                       for aa2 in df.columns}
        return matrix_dict, list(df.index)


    @staticmethod
    def load_blosum_matrix(matrix_file):
        try:
            with open(matrix_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                raise ValueError(f"The BLOSUM file '{matrix_file}' is empty.")

            # Find the line with amino acid order
            aa_order = None
            matrix_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    aa_order = line.split()
                    matrix_start = i + 1
                    break

            if aa_order is None:
                raise ValueError(f"Could not find amino acid order in the BLOSUM file '{matrix_file}'. "
                                 f"Please ensure the file is in the correct format.")

            matrix = {}
            for line in lines[matrix_start:]:
                if line.strip() and not line.startswith('#'):
                    values = line.split()
                    if len(values) != len(aa_order) + 1:
                        raise ValueError(f"Inconsistent number of values in line: {line}")
                    aa = values[0]
                    scores = list(map(int, values[1:]))
                    for other_aa, score in zip(aa_order, scores):
                        matrix[(aa, other_aa)] = score

            if not matrix:
                raise ValueError(f"No valid matrix data found in the BLOSUM file '{matrix_file}'.")

            return matrix, aa_order

        except FileNotFoundError:
            raise FileNotFoundError(f"The BLOSUM file '{matrix_file}' was not found.")
        except ValueError as e:
            raise ValueError(f"Error parsing BLOSUM file '{matrix_file}': {str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading the BLOSUM file '{matrix_file}': {str(e)}")

    @staticmethod
    def load_3di_matrix(matrix_file):
        try:
            with open(matrix_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                raise ValueError(f"The 3Di matrix file '{matrix_file}' is empty.")

            # Skip comment lines
            data_lines = [line for line in lines if not line.strip().startswith('#')]

            # First data line contains the alphabet
            alphabet_order = data_lines[0].split()

            matrix = {}
            for line in data_lines[1:]:  # Start from the second data line
                values = line.split()
                if len(values) != len(alphabet_order) + 1:
                    print(f"Warning: Inconsistent number of values in line: {line.strip()}")
                    continue
                char = values[0]
                scores = list(map(float, values[1:]))  # Convert scores to float
                for other_char, score in zip(alphabet_order, scores):
                    matrix[(char, other_char)] = score

            if not matrix:
                raise ValueError(f"No valid matrix data found in the 3Di matrix file '{matrix_file}'.")

            return matrix, alphabet_order

        except FileNotFoundError:
            raise FileNotFoundError(f"The 3Di matrix file '{matrix_file}' was not found.")
        except Exception as e:
            raise ValueError(f"Error parsing 3Di matrix file '{matrix_file}': {str(e)}")


    @staticmethod
    def calculate_similarity(score, max_score, min_score):
        # Normalize the score to a range of 0 to 1
        return (score - min_score) / (max_score - min_score)

    @staticmethod
    def calculate_pairwise_similarity(seq1, seq2, matrix, max_score, min_score):
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must be of equal length")

        similarities = []
        for aa1, aa2 in zip(seq1, seq2):
            if aa1 != '-' and aa2 != '-':
                score = matrix.get((aa1, aa2), matrix.get((aa2, aa1), min_score))
                similarity = AlignmentEvaluator.calculate_similarity(score, max_score, min_score)
                similarities.append(similarity)

        return similarities if similarities else [0.0]

    @staticmethod
    def calculate_residue_lddt(predicted_points, true_points, cutoff=15.0):
        dmat_true = np.sqrt(np.sum((true_points[:, None] - true_points[None, :]) ** 2, axis=-1))
        dmat_predicted = np.sqrt(np.sum((predicted_points[:, None] - predicted_points[None, :]) ** 2, axis=-1))

        dists_to_score = (dmat_true < cutoff).astype(np.float32) * (1 - np.eye(dmat_true.shape[0]))

        dist_l1 = np.abs(dmat_true - dmat_predicted)

        score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                        (dist_l1 < 1.0).astype(np.float32) +
                        (dist_l1 < 2.0).astype(np.float32) +
                        (dist_l1 < 4.0).astype(np.float32))

        norm = 1.0 / (1e-10 + np.sum(dists_to_score, axis=-1))
        lddt_scores = norm * (1e-10 + np.sum(dists_to_score * score, axis=-1))

        return lddt_scores


def evaluate_alignment_quality(pdb_files, fasta_file, matrix_file, matrix, max_score, min_score, aa_to_3di_map, _3di_matrix, lddt_cutoff=15.0, weights=None):
    if weights is None:
        weights = {'seq': 1, 'lddt': 1, '3di': 1}

    # Parse PDB files
    pdb_data = {}
    for pdb_file in pdb_files:
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
        seq, coords = CustomPDBParser.parse_pdb_sequence_and_coordinates(pdb_file)
        pdb_data[pdb_name] = {'seq': seq, 'coords': coords}

    # Parse FASTA alignment
    aligned_sequences = FASTAParser.parse_fasta_alignment(fasta_file)

    # Match PDB sequences with FASTA sequences
    if set(pdb_data.keys()) != set(aligned_sequences.keys()):
        raise ValueError("PDB file names do not match sequence names in FASTA file")

    # Calculate pairwise similarities and LDDT scores
    results = []
    for (name1, name2) in combinations(aligned_sequences.keys(), 2):
        seq1 = aligned_sequences[name1]
        seq2 = aligned_sequences[name2]
        _3di_seq1 = aa_to_3di(seq1, name1, aa_to_3di_map)
        _3di_seq2 = aa_to_3di(seq2, name2, aa_to_3di_map)
        coords1 = pdb_data[name1]['coords']
        coords2 = pdb_data[name2]['coords']

        seq_similarities = AlignmentEvaluator.calculate_pairwise_similarity(seq1, seq2, matrix, max_score, min_score)
        _3di_similarities = AlignmentEvaluator.calculate_pairwise_similarity(_3di_seq1, _3di_seq2, _3di_matrix, max(_3di_matrix.values()), min(_3di_matrix.values()))

        aligned_coords1 = []
        aligned_coords2 = []
        pdb_pos1 = 0
        pdb_pos2 = 0

        combined_scores = []
        for aln_pos, (res1, res2) in enumerate(zip(seq1, seq2)):
            if res1 != '-' and res2 != '-':
                pdb_pos1 += 1
                pdb_pos2 += 1
                if pdb_pos1 <= len(coords1) and pdb_pos2 <= len(coords2):
                    aligned_coords1.append(coords1[pdb_pos1]['ca'])
                    aligned_coords2.append(coords2[pdb_pos2]['ca'])

                    seq_sim = seq_similarities[len(combined_scores)]
                    _3di_sim = _3di_similarities[len(combined_scores)]

                    combined_scores.append({
                        'aln_pos': aln_pos + 1,
                        'residue_pdb1': coords1[pdb_pos1]['residue'],
                        'residue_pdb2': coords2[pdb_pos2]['residue'],
                        'seq_similarity': seq_sim,
                        '3di_similarity': _3di_sim,
                    })
            elif res1 != '-':
                pdb_pos1 += 1
            elif res2 != '-':
                pdb_pos2 += 1

        # LDDT calculation
        aligned_coords1 = np.array(aligned_coords1)
        aligned_coords2 = np.array(aligned_coords2)
        lddt_scores = AlignmentEvaluator.calculate_residue_lddt(aligned_coords1, aligned_coords2, lddt_cutoff)

        # Add LDDT scores and calculate average scores
        for i, score in enumerate(combined_scores):
            lddt = lddt_scores[i]
            score['lddt_score'] = lddt
            score['average_score'] = (score['seq_similarity'] * weights['seq'] +
                                      lddt * weights['lddt'] +
                                      score['3di_similarity'] * weights['3di']) / sum(weights.values())

        # Calculate global scores
        global_seq_sim = np.mean([score['seq_similarity'] for score in combined_scores])
        global_lddt = np.mean(lddt_scores)
        global_3di = np.mean([score['3di_similarity'] for score in combined_scores])
        global_average = (global_seq_sim * weights['seq'] +
                          global_lddt * weights['lddt'] +
                          global_3di * weights['3di']) / sum(weights.values())

        results.append({
            'pair': (name1, name2),
            'scores': combined_scores,
            'global_seq_sim': global_seq_sim,
            'global_lddt': global_lddt,
            'global_3di': global_3di,
            'global_average': global_average
        })

    return results


def score_alignment(alignment, pdb_dir,
                    aa_matrix, aa_max_score, aa_min_score,
                    _3di_matrix, _3di_max_score, _3di_min_score,
                    aa_to_3di_map):
    print(f"Debug: Entering score_alignment function")
    print(f"Debug: pdb_dir = {pdb_dir}")

    if len(alignment) < 2:
        print(f"Warning: Alignment has fewer than 2 sequences. Returning default score.")
        return {'seq_sim': 0.0, 'lddt': 0.0, '3di': 0.0, 'average': 0.0}

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_file:
        AlignIO.write(alignment, temp_file, "fasta")
        temp_file_path = temp_file.name

    try:
        sequence_ids = [record.id for record in alignment]
        if len(set(sequence_ids)) == 1:  # All sequences are identical
            print(f"Debug: All sequences are identical. Returning perfect score.")
            return {'seq_sim': 1.0, 'lddt': 1.0, '3di': 1.0, 'average': 1.0}

        pdb_files = get_pdb_files_from_dir(pdb_dir, sequence_ids)

        print(f"Debug: pdb_files = {pdb_files}")

        results = evaluate_alignment_quality(pdb_files, temp_file_path, None, aa_matrix, aa_max_score, aa_min_score, aa_to_3di_map, _3di_matrix)

        # Extract individual scores
        seq_sim = results[0]['global_seq_sim']
        lddt = results[0]['global_lddt']
        _3di = results[0]['global_3di']
        average = results[0]['global_average']

        print(f"Debug: Sequence similarity score = {seq_sim}")
        print(f"Debug: LDDT score = {lddt}")
        print(f"Debug: 3Di score = {_3di}")
        print(f"Debug: Average score = {average}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {'seq_sim': float('-inf'), 'lddt': float('-inf'), '3di': float('-inf'), 'average': float('-inf')}
    except Exception as e:
        print(f"Error evaluating alignment: {e}")
        return {'seq_sim': float('-inf'), 'lddt': float('-inf'), '3di': float('-inf'), 'average': float('-inf')}
    finally:
        os.remove(temp_file_path)

    print(f"Debug: Exiting score_alignment function")
    return {'seq_sim': seq_sim, 'lddt': lddt, '3di': _3di, 'average': average}


def score_multiple_alignment(alignment, pdb_dir, matrix_file, _3di_matrix_file, aa_to_3di_map,
                             aa_matrix, aa_max_score, aa_min_score,
                             _3di_matrix, _3di_max_score, _3di_min_score):
    print(f"Debug: Entering score_multiple_alignment function")
    print(f"Debug: pdb_dir = {pdb_dir}")
    print(f"Debug: matrix_file = {matrix_file}")
    print(f"Debug: _3di_matrix_file = {_3di_matrix_file}")

    if len(alignment) < 2:
        print(f"Warning: Alignment has fewer than 2 sequences. Returning default score.")
        return {'seq_sim': 0.0, 'lddt': 0.0, '3di': 0.0, 'average': 0.0}

    try:
        sequence_ids = [record.id for record in alignment]
        pdb_files = get_pdb_files_from_dir(pdb_dir, sequence_ids)

        # Load BLOSUM matrix
        matrix, aa_order = AlignmentEvaluator.load_blosum_matrix(matrix_file)
        max_score = max(matrix.values())
        min_score = min(matrix.values())

        # Load 3Di matrix
        _3di_matrix, _3di_alphabet = AlignmentEvaluator.load_3di_matrix(_3di_matrix_file)

        # Parse PDB files
        pdb_data = {}
        for pdb_file in pdb_files:
            pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
            seq, coords = CustomPDBParser.parse_pdb_sequence_and_coordinates(pdb_file)
            pdb_data[pdb_name] = {'seq': seq, 'coords': coords}

        # Calculate pairwise similarities and LDDT scores
        total_seq_sim = 0
        total_lddt = 0
        total_3di = 0
        pair_count = 0

        for i, seq1 in enumerate(alignment):
            for j, seq2 in enumerate(alignment[i + 1:], start=i + 1):
                seq_similarities = []
                _3di_similarities = []
                aligned_coords1 = []
                aligned_coords2 = []

                _3di_seq1 = aa_to_3di(str(seq1.seq), seq1.id, aa_to_3di_map)
                _3di_seq2 = aa_to_3di(str(seq2.seq), seq2.id, aa_to_3di_map)

                # Only consider positions where all sequences have non-gap characters
                for k, (res1, res2) in enumerate(zip(seq1.seq, seq2.seq)):
                    if res1 != '-' and res2 != '-' and all(seq[k] != '-' for seq in alignment):
                        score = matrix.get((res1, res2), matrix.get((res2, res1), min_score))
                        similarity = AlignmentEvaluator.calculate_similarity(score, max_score, min_score)
                        seq_similarities.append(similarity)

                        _3di_score = _3di_matrix.get((_3di_seq1[k], _3di_seq2[k]), _3di_matrix.get((_3di_seq2[k], _3di_seq1[k]), min(_3di_matrix.values())))
                        _3di_similarity = AlignmentEvaluator.calculate_similarity(_3di_score, max(_3di_matrix.values()), min(_3di_matrix.values()))
                        _3di_similarities.append(_3di_similarity)

                        aligned_coords1.append(pdb_data[seq1.id]['coords'][len(aligned_coords1) + 1]['ca'])
                        aligned_coords2.append(pdb_data[seq2.id]['coords'][len(aligned_coords2) + 1]['ca'])

                if seq_similarities:
                    total_seq_sim += np.mean(seq_similarities)
                    total_3di += np.mean(_3di_similarities)

                    aligned_coords1 = np.array(aligned_coords1)
                    aligned_coords2 = np.array(aligned_coords2)
                    lddt_scores = AlignmentEvaluator.calculate_residue_lddt(aligned_coords1, aligned_coords2)
                    total_lddt += np.mean(lddt_scores)

                    pair_count += 1

        if pair_count == 0:
            print(f"Warning: No valid pairs found in alignment. Returning default score.")
            return {'seq_sim': 0.0, 'lddt': 0.0, '3di': 0.0, 'average': 0.0}

        avg_seq_sim = total_seq_sim / pair_count
        avg_lddt = total_lddt / pair_count
        avg_3di = total_3di / pair_count

        # You can adjust these weights as needed
        weights = {'seq': 1, 'lddt': 1, '3di': 1}
        final_score = (avg_seq_sim * weights['seq'] + avg_lddt * weights['lddt'] + avg_3di * weights['3di']) / sum(weights.values())

        print(f"Debug: Average sequence similarity = {avg_seq_sim}")
        print(f"Debug: Average LDDT score = {avg_lddt}")
        print(f"Debug: Average 3Di score = {avg_3di}")
        print(f"Debug: Final alignment score = {final_score}")

        return {'seq_sim': avg_seq_sim, 'lddt': avg_lddt, '3di': avg_3di, 'average': final_score}

    except Exception as e:
        print(f"Error evaluating multiple alignment: {e}")
        return {'seq_sim': float('-inf'), 'lddt': float('-inf'), '3di': float('-inf'), 'average': float('-inf')}


@lru_cache(maxsize=None)
def cached_score_alignment(alignment_key, pdb_dir,
                           aa_matrix_frozen, aa_max_score, aa_min_score,
                           _3di_matrix_frozen, _3di_max_score, _3di_min_score,
                           aa_to_3di_map_frozen):
    # Convert frozen sets back to dictionaries
    aa_matrix = dict(aa_matrix_frozen)
    _3di_matrix = dict(_3di_matrix_frozen)
    aa_to_3di_map = frozenset_to_dict(aa_to_3di_map_frozen)

    # Convert the key back to an alignment object
    alignment = MultipleSeqAlignment([SeqRecord(Seq(seq), id=id) for id, seq in alignment_key])

    if len(alignment) > 2:
        return score_multiple_alignment(alignment, pdb_dir,
                                        aa_matrix, aa_max_score, aa_min_score,
                                        _3di_matrix, _3di_max_score, _3di_min_score,
                                        aa_to_3di_map)
    else:
        return score_alignment(alignment, pdb_dir,
                               aa_matrix, aa_max_score, aa_min_score,
                               _3di_matrix, _3di_max_score, _3di_min_score,
                               aa_to_3di_map)

def find_best_leaf_alignment(leaf_node, parent_node, leaf_dict, pdb_dir,
                             aa_matrix, aa_max_score, aa_min_score,
                             _3di_matrix, _3di_max_score, _3di_min_score,
                             aa_to_3di_map):
    leaf_descendants = set(parent_node.name.split('_'))
    best_alignment = None
    best_score = {'seq_sim': float('-inf'), 'lddt': float('-inf'), '3di': float('-inf'), 'average': float('-inf')}

    for key, alignment in leaf_dict.items():
        if leaf_node.name in key and any(leaf in key for leaf in leaf_descendants if leaf != leaf_node.name):
            alignment, _ = remove_duplicates(alignment)  # Remove duplicates

            if len(alignment) > 2:
                score = score_multiple_alignment(alignment, pdb_dir,
                                                 aa_matrix, aa_max_score, aa_min_score,
                                                 _3di_matrix, _3di_max_score, _3di_min_score,
                                                 aa_to_3di_map)
            else:
                alignment_key = tuple((rec.id, str(rec.seq)) for rec in alignment)
                score = cached_score_alignment(alignment_key, pdb_dir,
                                               dict_to_frozenset(aa_matrix), aa_max_score, aa_min_score,
                                               dict_to_frozenset(_3di_matrix), _3di_max_score, _3di_min_score,
                                               dict_to_frozenset(aa_to_3di_map))

            if score['average'] > best_score['average']:
                best_score = score
                best_alignment = alignment

    return best_alignment


"""
************************************************************************************************************************
************************************************* RECURSIVE ALIGNMENT **************************************************
************************************************************************************************************************
"""


def recursive_alignment(node, node_classifications, leaf_dict, node_dict, pdb_dir,
                        aa_matrix, aa_max_score, aa_min_score,
                        _3di_matrix, _3di_max_score, _3di_min_score,
                        aa_to_3di_map, _3di_matrix_file, matrix_file, level=0):
    indent = "  " * level
    node_type = node_classifications[node.name]
    print(f"{indent}Processing node: {node.name} (Type: {node_type})")

    # Log the current node's children
    print(f"{indent}Children of {node.name}: {[child.name for child in node.children]}")

    if node_type == "leaf":
        print(f"{indent}Leaf node detected: {node.name}")
        return None

    elif node_type == "leaf_binary":
        print(f"{indent}Processing leaf binary node: {node.name}")
        alignment = leaf_dict.get(tuple(node.name.split('_')))
        if alignment is None:
            print(f"{indent}No alignment found for leaf binary node: {node.name}")
            return None
        print(f"{indent}Alignment found for {node.name}. Sequences: {[rec.id for rec in alignment]}")
        alignment, removed = remove_duplicates(alignment)
        if removed:
            print(f"{indent}Removed duplicate sequences: {', '.join(removed)}")
        alignment_key = tuple((rec.id, str(rec.seq)) for rec in alignment)
        scores = cached_score_alignment(alignment_key, pdb_dir,
                                        frozenset(aa_matrix.items()), aa_max_score, aa_min_score,
                                        frozenset(_3di_matrix.items()), _3di_max_score, _3di_min_score,
                                        dict_to_frozenset(aa_to_3di_map))
        print(f"{indent}Leaf binary node, alignment scores:")
        print(f"{indent}  Sequence similarity: {scores['seq_sim']:.4f}")
        print(f"{indent}  LDDT score: {scores['lddt']:.4f}")
        print(f"{indent}  3Di score: {scores['3di']:.4f}")
        print(f"{indent}  Average score: {scores['average']:.4f}")
        print_node_alignment(node.name, alignment, level)
        return alignment

    elif node_type in ["leaf_complex", "node_binary", "node_complex", "mixed_complex"]:
        print(f"{indent}Processing {node_type} node: {node.name}")
        child_alignments = []
        for child in node.children:
            print(f"{indent}Processing child node: {child.name}")
            child_result = recursive_alignment(child, node_classifications, leaf_dict, node_dict, pdb_dir,
                                               aa_matrix, aa_max_score, aa_min_score,
                                               _3di_matrix, _3di_max_score, _3di_min_score,
                                               aa_to_3di_map, _3di_matrix_file, matrix_file, level + 1)
            if child_result is not None:
                child_alignments.append(child_result)
                print(f"{indent}Added alignment for {child.name}. Sequences: {[rec.id for rec in child_result]}")
            elif node_classifications[child.name] == "leaf":
                print(f"{indent}Searching for best leaf alignment for: {child.name}")
                leaf_alignment = find_best_leaf_alignment(child, node, leaf_dict, pdb_dir,
                                                          aa_matrix, aa_max_score, aa_min_score,
                                                          _3di_matrix, _3di_max_score, _3di_min_score,
                                                          aa_to_3di_map)
                if leaf_alignment:
                    child_alignments.append(leaf_alignment)
                    print(
                        f"{indent}Selected best alignment for leaf: {child.name}. Sequences: {[rec.id for rec in leaf_alignment]}")
                else:
                    print(f"{indent}No suitable alignment found for leaf: {child.name}")

        if not child_alignments:
            print(f"{indent}No valid child alignments found for {node.name}")
            return None

        print(f"{indent}Combining alignments for {node_type} node: {node.name}")
        print(f"{indent}Number of child alignments: {len(child_alignments)}")
        for i, aln in enumerate(child_alignments):
            print(f"{indent}Child alignment {i + 1} sequences: {[rec.id for rec in aln]}")

        result_alignment = combine_alignments_mafft(child_alignments, aa_to_3di_map, _3di_matrix_file)

        print(f"{indent}Combined alignment for {node.name}. Sequences: {[rec.id for rec in result_alignment]}")

        result_alignment, removed = remove_duplicates(result_alignment)
        if removed:
            print(f"{indent}Removed duplicate sequences after combining: {', '.join(removed)}")

        final_scores = score_multiple_alignment(result_alignment, pdb_dir,
                                                matrix_file, _3di_matrix_file, aa_to_3di_map,
                                                aa_matrix, aa_max_score, aa_min_score,
                                                _3di_matrix, _3di_max_score, _3di_min_score)
        print(f"{indent}{node_type.capitalize()} node, final scores:")
        print(f"{indent}  Sequence similarity: {final_scores['seq_sim']:.4f}")
        print(f"{indent}  LDDT score: {final_scores['lddt']:.4f}")
        print(f"{indent}  3Di score: {final_scores['3di']:.4f}")
        print(f"{indent}  Average score: {final_scores['average']:.4f}")
        print_node_alignment(node.name, result_alignment, level)
        return result_alignment

    return None

"""
************************************************************************************************************************
******************************************************** UTILS *********************************************************
************************************************************************************************************************
"""

def load_scoring_matrices(matrix_dir):
    # Set up the correct paths for matrix files
    aa_matrix_file = matrix_dir / "BLOSUM62.txt"
    _3di_matrix_file = matrix_dir / "3di_matrix.txt"

    # Check if matrix files exist
    if not aa_matrix_file.exists():
        raise FileNotFoundError(f"AA matrix file not found: {aa_matrix_file}")

    if not _3di_matrix_file.exists():
        raise FileNotFoundError(f"3Di matrix file not found: {_3di_matrix_file}")

    # Load matrices
    aa_matrix, aa_alphabet = AlignmentEvaluator.load_blosum_matrix(str(aa_matrix_file))
    _3di_matrix, _3di_alphabet = AlignmentEvaluator.load_3di_matrix(str(_3di_matrix_file))

    # Calculate max and min scores for both matrices
    aa_max_score = max(aa_matrix.values())
    aa_min_score = min(aa_matrix.values())
    _3di_max_score = max(_3di_matrix.values())
    _3di_min_score = min(_3di_matrix.values())

    return {
        'aa_matrix': aa_matrix,
        'aa_alphabet': aa_alphabet,
        'aa_max_score': aa_max_score,
        'aa_min_score': aa_min_score,
        '_3di_matrix': _3di_matrix,
        '_3di_alphabet': _3di_alphabet,
        '_3di_max_score': _3di_max_score,
        '_3di_min_score': _3di_min_score
    }

def print_node_alignment(node_name, alignment, level=0):
    indent = "  " * level
    print(f"\n{indent}Alignment for node: {node_name}")
    print_alignment(alignment, max_name_length=20, line_length=60)


def remove_duplicates(alignment):
    seen = set()
    unique_alignment = MultipleSeqAlignment([])
    removed_sequences = []

    for record in alignment:
        if record.id not in seen:
            seen.add(record.id)
            unique_alignment.append(record)
        else:
            removed_sequences.append(record.id)

    return unique_alignment, removed_sequences


def pad_sequence(sequence, length):
    """Pad a sequence to the given length."""
    return sequence + '-' * (length - len(sequence))


def combine_alignments(aln1, aln2):
    """Combine two alignments vertically and realign using MAFFT."""
    # Find the maximum length among all sequences
    max_length = max(aln1.get_alignment_length(), aln2.get_alignment_length())

    # Pad sequences in aln1
    padded_aln1 = MultipleSeqAlignment([
        SeqRecord(Seq(str(rec.seq).ljust(max_length, '-')), id=rec.id, description="")
        for rec in aln1
    ])

    # Pad sequences in aln2
    padded_aln2 = MultipleSeqAlignment([
        SeqRecord(Seq(str(rec.seq).ljust(max_length, '-')), id=rec.id, description="")
        for rec in aln2
    ])

    # Combine padded sequences from both alignments
    combined = MultipleSeqAlignment([])
    combined.extend(padded_aln1)
    combined.extend([rec for rec in padded_aln2 if rec.id not in [seq.id for seq in padded_aln1]])

    # Write combined sequences to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_input_file:
        AlignIO.write(combined, temp_input_file, "fasta")
        temp_input_path = temp_input_file.name

    try:
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta') as temp_output_file:
            temp_output_path = temp_output_file.name

        # Run MAFFT alignment
        mafft_output = run_mafft_alignment(temp_input_path, temp_output_path)

        return mafft_output
    except Exception as e:
        print(f"Error during MAFFT alignment: {e}")
        return combined  # Return the original combined alignment if MAFFT fails
    finally:
        os.remove(temp_input_path)
        if 'temp_output_path' in locals():
            os.remove(temp_output_path)


def get_pdb_files_from_dir(pdb_dir, sequence_ids):
    """
    Convert a directory path to a list of PDB file paths based on sequence IDs.

    Args:
    pdb_dir (str): Path to the directory containing PDB files
    sequence_ids (list): List of sequence IDs to look for

    Returns:
    list: List of PDB file paths
    """
    pdb_files = []
    for seq_id in sequence_ids:
        pdb_file = os.path.join(pdb_dir, f"{seq_id}.pdb")
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        pdb_files.append(pdb_file)
    return pdb_files


def print_alignment(alignment, max_name_length=20, line_length=60):
    """
    Print the alignment in a readable format.

    Args:
    alignment (MultipleSeqAlignment): The alignment to print
    max_name_length (int): Maximum length for sequence names
    line_length (int): Number of characters to print per line
    """
    if not alignment:
        print("No alignment to print.")
        return

    seq_length = alignment.get_alignment_length()

    for start in range(0, seq_length, line_length):
        end = min(start + line_length, seq_length)
        print(f"\nPositions {start + 1}-{end}:")

        for record in alignment:
            name = record.id[:max_name_length].ljust(max_name_length)
            seq_slice = record.seq[start:end]
            print(f"{name} {seq_slice}")
        print()  # Empty line between blocks


def dict_to_frozenset(d):
    if isinstance(d, dict):
        return frozenset((k, dict_to_frozenset(v)) for k, v in d.items())
    elif isinstance(d, (list, tuple)):
        return tuple(dict_to_frozenset(v) for v in d)
    else:
        return d

def frozenset_to_dict(f):
    if isinstance(f, frozenset):
        return {k: frozenset_to_dict(v) for k, v in f}
    elif isinstance(f, tuple):
        return tuple(frozenset_to_dict(v) for v in f)
    else:
        return f


"""
************************************************************************************************************************
**************************************************** MAIN WORKFLOW *****************************************************
************************************************************************************************************************
"""
node_class_dict = {
    "leaf": "A terminal node representing a single sequence",
    "leaf_binary": "A node with two leaf children",
    "node_binary": "A node with two non-leaf children",
    "leaf_complex": "A node with more than two leaf children",
    "node_complex": "A node with more than two non-leaf children",
    "mixed_complex": "A node with a mix of leaf and non-leaf children"
}


def main():
    # Set up the correct path for the matrix directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    MATRIX_DIR = SCRIPT_DIR / "scoring_matrix"

    # Ensure MATRIX_DIR exists
    if not MATRIX_DIR.exists():
        raise FileNotFoundError(f"Matrix directory not found: {MATRIX_DIR}")

    # Load matrices and scores
    matrix_data = load_scoring_matrices(MATRIX_DIR)

    # Define paths
    tree_file_path = "/home/casey/Desktop/lab_projects/test/matrix2tree/fident_matrix_ln_rooted/rooted_upgma_tree.tree"
    fasta_folder_path = "/home/casey/Desktop/lab_projects/test/pdb2pairwise/usalign_fNS/pairwise_fastas"
    pdb_dir = "/home/casey/Desktop/lab_projects/foldMSA_USalign/test/pdbs"
    matrix_file = MATRIX_DIR / "BLOSUM62.txt"
    _3di_matrix_file = MATRIX_DIR / "3di_matrix.txt"

    # Build AA to 3Di mapping
    aa_to_3di_map = build_aa_to_3di_mapping(pdb_dir)

    # Process the tree
    labeled_tree = process_tree(tree_file_path)
    leaf_dict = load_pairwise_fastas(fasta_folder_path)
    node_dict = {}

    # Print some information about the loaded data
    print(f"\nLoaded {len(leaf_dict)} pairwise alignments.")
    for key, alignment in list(leaf_dict.items())[:5]:  # Print details of first 5 alignments
        print(f"Alignment for {key}: {len(alignment)} sequences, {alignment.get_alignment_length()} positions")

    print(f"\nInitialized empty node_dict. Current size: {len(node_dict)}")

    # Traverse the labeled tree and check for matches
    print("\nTraversing labeled tree and checking for matches:")
    if labeled_tree and leaf_dict:
        traverse_and_match(labeled_tree, leaf_dict, node_dict)

    # Run the classification and visualization
    print("\nClassifying nodes in the tree:")
    if labeled_tree:
        node_classifications = classify_nodes(labeled_tree)

        print("\nNode Classifications:")
        for node_name, classification in node_classifications.items():
            print(f"{node_name}: {classification}")

        print("\nNode Classification Descriptions:")
        for class_name, description in node_class_dict.items():
            print(f"{class_name}: {description}")

        print_ete_tree(labeled_tree)
    else:
        print("No labeled tree available. Please ensure the tree is properly loaded and labeled.")

    # You can now use node_classifications dictionary for further processing
    print("\nnode_classifications dictionary:", node_classifications)

    # Verify that the paths are correct
    print(f"Debug: pdb_dir = {pdb_dir}")
    print(f"Debug: matrix_file = {matrix_file}")
    print(f"Debug: _3di_matrix_file = {_3di_matrix_file}")

    if not os.path.isdir(pdb_dir):
        raise ValueError(f"PDB directory not found: {pdb_dir}")

    final_result = recursive_alignment(labeled_tree, node_classifications, leaf_dict, node_dict, pdb_dir,
                                       matrix_data['aa_matrix'], matrix_data['aa_max_score'], matrix_data['aa_min_score'],
                                       matrix_data['_3di_matrix'], matrix_data['_3di_max_score'], matrix_data['_3di_min_score'],
                                       aa_to_3di_map, str(_3di_matrix_file), str(matrix_file))

    if final_result:
        # Perform a final MAFFT alignment
        final_alignment = final_mafft_alignment(final_result, aa_to_3di_map, str(_3di_matrix_file))

        # Final duplicate removal
        final_alignment, removed = remove_duplicates(final_alignment)
        if removed:
            print(f"\nRemoved duplicate sequences in final alignment: {', '.join(removed)}")

        # Ensure all six sequences are in the alignment
        expected_sequences = {'btAID', 'hsAID', 'hsA3A', 'saA3A', 'mmAID', 'ddd1'}
        missing_sequences = expected_sequences - set(seq.id for seq in final_alignment)
        if missing_sequences:
            print(f"Warning: Missing sequences in final alignment: {', '.join(missing_sequences)}")
            # Here you might want to add code to retrieve and add the missing sequences

        # Score the final alignment
        final_score = score_multiple_alignment(final_alignment, pdb_dir,
                                               str(matrix_file), str(_3di_matrix_file), aa_to_3di_map,
                                               matrix_data['aa_matrix'], matrix_data['aa_max_score'],
                                               matrix_data['aa_min_score'],
                                               matrix_data['_3di_matrix'], matrix_data['_3di_max_score'],
                                               matrix_data['_3di_min_score'])

        print("\nFinal alignment details:")
        print(f"Number of sequences: {len(final_alignment)}")
        print(f"Alignment length: {final_alignment.get_alignment_length()}")
        print(f"Sequence IDs: {', '.join(seq.id for seq in final_alignment)}")
        print("Final scores:")
        for key, value in final_score.items():
            print(f"  {key}: {value:.4f}")
        print("\nFinal Alignment:")
        print_alignment(final_alignment)
    else:
        print("Failed to generate final alignment")

if __name__ == "__main__":
    main()