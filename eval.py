import numpy as np
import re
import os

# This script evaluates all log files in the output folder by calculating the mean and the
# standard deviation of all three seeds of one run.

current_dir = os.getcwd()
data_path = "/output/Caltech/"


def main() -> None:
    folder_path = current_dir + data_path
    print("Test and train accuracy with standard deviation of different runs.")
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for file in files:
            eval_file(file)
    else:
        print(f"The folder {folder_path} does not exist")


def eval_file(file: str) -> None:
    # Open the log file for reading
    full_path = current_dir + data_path + file
    test_mean, test_std_dev = calc_acc_and_std_dev(full_path, "* accuracy", r'\d+\.\d+', False)
    train_mean, train_std_dev = calc_acc_and_std_dev(full_path, "epoch [", r'acc (\d+\.\d+)', True)

    # Print the results
    print(f"{file} test: {test_mean}% +- {test_std_dev}% and train: {train_mean}% +- {train_std_dev}")


def calc_acc_and_std_dev(path: str, line_pattern: str, reg_pattern: str, switch: bool) -> tuple[float, float]:
    lines_with_accuracy = []
    accuracies = []
    with open(path, 'r') as log_file:
        for line in log_file:
            # Check if the line contains "* accuracy"
            if line_pattern in line:
                # Append the line to the list
                lines_with_accuracy.append(line)
    # Now, lines_with_accuracy contains all the lines with "* accuracy"
    for line in lines_with_accuracy:
        # Use a regular expression to extract the numeric value
        match = re.search(reg_pattern, line)
        if match:
            # Convert the matched value to a float
            if switch:
                accuracy = float(match.group(1))
            else:
                accuracy = float(match.group())
            accuracies.append(accuracy)
        else:
            print("No numeric value found.")
    # Calculate the test mean
    test_mean = round(np.mean(accuracies), 1)
    # Calculate the test standard deviation
    test_std_dev = round(np.std(accuracies), 1)
    return test_mean, test_std_dev


if __name__ == "__main__":
    main()
