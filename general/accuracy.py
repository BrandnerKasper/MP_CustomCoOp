import numpy as np
import re
import os
# Open the log file for reading
current_dir = os.getcwd()
data_path = "/output/Caltech/"
file_name = 'open_clip_RN101'
full_path = current_dir + data_path + file_name + '.txt'
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
std_dev = round(np.std(accuracies),1)

# Print the results
print(f"{file_name} Mean with Standard Deviation: {mean}% +- {std_dev}%")
