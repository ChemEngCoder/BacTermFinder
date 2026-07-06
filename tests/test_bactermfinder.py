import subprocess
import os
from pathlib import Path
import pytest


def validate_expected_output(expected_path, temp_file_path):
    # Validate contents of output files
    with open(temp_file_path, "r") as f_actual, open(expected_path, "r") as f_expected:
        actual_lines = [line.strip() for line in f_actual if line.strip() and not line.startswith("#")]
        expected_lines = [line.strip() for line in f_expected if line.strip() and not line.startswith("#")]

    # Check that the number of predicted features matches exactly
    assert len(actual_lines) == len(expected_lines), (
        f"Mismatched number of lines. Expected {len(expected_lines)}, got {len(actual_lines)}."
    )

    # Line-by-line comparison for structural and data consistency
    for idx, (act, exp) in enumerate(zip(actual_lines, expected_lines)):
        assert act == exp, f"Row mismatch at data line {idx + 1}.\nExpected: {exp}\nGot: {act}"


# --- The Main Test ---
def test_script_execution(tmp_path):
    """
    Tests the external script by passing paths and verifying the generated files.
    """
    # 1. Setup paths using the tmp_path fixture
    root_dir = Path(__file__).parent
    script_path = os.path.join(root_dir, "genome_scan.py")
    test_filename = "ref-genome-test.fasta"
    test_path = os.path.join(root_dir, test_filename)
    tmp_dir = tmp_path / "output_results"  # Script will create this

    # 2. Create paths for expected outputs
    temp_data_prefixes = ["_ENAC", "_NCP", "_PS2", "_binary"]
    temp_data_suffixes = ["mean.csv", "sliding_windows.csv"] # Removed "result.csv"
    result_files = {
        "_binary": "out_binaryref-genome-test-CP054306.1-Synechocystis-PCC-7338.fasta.csv",
        "_ENAC": "out_ENACref-genome-test-CP054306.1-Synechocystis-PCC-7338.fasta.csv",
        "_NCP": "out_NCPref-genome-test-CP054306.1-Synechocystis-PCC-7338.fasta.csv",
        "_PS2": "out_PS2ref-genome-test-CP054306.1-Synechocystis-PCC-7338.fasta.csv",
        "sliding_windows.csv": "out_ref-genome-test-CP054306.1-Synechocystis-PCC-7338.fasta_sliding_windows.csv",
        "mean.csv" : "outref-genome-test-CP054306.1-Synechocystis-PCC-7338.fasta_mean.csv" 
    }

    # 3. Construct the command line arguments
    script_cmd = [
        "python", str(script_path), 
        "--fasta", str(test_path), 
        "--output-dir", str(tmp_dir),
        "--repo-dir", str(root_dir), 
        "--step-size", "3",
        "--batch-size", "10000",
        "--min-score", "0.3" 
    ]

    # 3. Execute the script
    script_result = subprocess.run(script_cmd, capture_output=True, text=True)

    # 4. Assertions: Check execution success
    assert script_result.returncode == 0, f"Script failed with stderr: {script_result.stderr}"

    # 5. Assertions: Check file presence
    for temp_prefix in temp_data_prefixes:
        temp_file = f"{temp_prefix}{test_filename}.csv"
        temp_file_path = os.path.join(tmp_dir, temp_file)
        assert os.path.exists(temp_file_path), f"Expected output file {temp_file_path} was not created."
        assert os.stat(temp_file_path).st_size > 0, "The generated temp file is empty."

        # Validate contents of output files
        expected_filename = result_files[temp_prefix]
        expected_path = os.path.join(tmp_dir, expected_filename)

        validate_expected_output(expected_path, temp_file_path)


    for temp_suffix in temp_data_suffixes:
        if temp_suffix == "sliding_windows.csv":
            temp_file = f"_{test_filename}_{temp_suffix}"
        else:
            temp_file = f"{test_filename}_{temp_suffix}"
        temp_file_path = os.path.join(tmp_dir, temp_file)
        assert os.path.exists(temp_file_path), f"Expected output file {temp_file_path} was not created."
        assert os.stat(temp_file_path).st_size > 0, "The generated temp file is empty."

        # Validate contents of output files
        expected_filename = result_files[temp_prefix]
        expected_path = os.path.join(tmp_dir, expected_filename)

        validate_expected_output(expected_path, temp_file_path)

    