import numpy as np
import re
import os

# This script evaluates all log files in the output folder by calculating the mean and the
# standard deviation of all three seeds of one run.

current_dir = os.getcwd()
data_path = "/output/Caltech/"


def main() -> None:
    folder_path = current_dir + data_path

    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for file in files:
            eval_file(file)
    else:
        print(f"The folder {folder_path} does not exist")


def eval_file(file: str) -> None:
    # Open the log file for reading
    full_path = current_dir + data_path + file
    lines_with_accuracy = []
    accuracies = []

    with open(full_path, 'r') as log_file:
        for line in log_file:
            # Check if the line contains "* accuracy"
            if "* accuracy" in line:
                # Append the line to the list
                lines_with_accuracy.append(line)

    # Now, lines_with_accuracy contains all the lines with "* accuracy"
    for line in lines_with_accuracy:
        # Use a regular expression to extract the numeric value
        match = re.search(r'\d+\.\d+', line)
        if match:
            # Convert the matched value to a float
            accuracy = float(match.group())
            accuracies.append(accuracy)
        else:
            print("No numeric value found.")

    # Calculate the mean
    mean = round(np.mean(accuracies), 1)

    # Calculate the standard deviation
    std_dev = round(np.std(accuracies), 1)

    # Print the results
    print(f"{file} Mean with Standard Deviation: {mean}% +- {std_dev}%")


if __name__ == "__main__":
    main()
