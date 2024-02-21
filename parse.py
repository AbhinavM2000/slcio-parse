import re

######---START PARSING----#####

def process_line(line):
    # Replace special characters with ','
    line = re.sub(r"[^\w\s.,+-]", ",", line)
    # Remove unnecessary spaces
    line = " ".join(line.split())
    return line


def remove_extra_commas(lines):
    cleaned_lines = []
    for line in lines:
        if line[:2] == "-,":
            line = line[2:]
        parts = line.split(",")
        cleaned_parts = [part.strip() for part in parts if part.strip()]
        cleaned_line = ",".join(cleaned_parts)
        cleaned_lines.append(cleaned_line)
    return cleaned_lines


def process_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    processed_lines = [process_line(line) for line in lines]
    cleaned_lines = remove_extra_commas(processed_lines)

    with open(output_file, "w") as f:
        for line in cleaned_lines:
            f.write(line + "\n")


# calling function
process_file("data.txt", "final_result.txt")

# print number of lines in final_result.txt
with open("final_result.txt", "r") as f:
    lines = f.readlines()
    print(len(lines))

import re

# open final_result.txt and if any line ends with +n, where n greater than 3, calculate sum of all such n = count. Then print % = count / (sum of all n including those less than 3)
with open("final_result.txt", "r") as f:
    lines = f.readlines()

sum_n = 0
count = 0
for line in lines:
    match = re.search(r"\+(\d+)$", line.strip())  # Check if the line ends with '+n'
    if match:
        n = int(match.group(1))
        if n > 3:  # Ensure n is greater than 3
            sum_n += n
            count += 1

total_sum = sum_n
for line in lines:
    match = re.search(r"\+(\d+)$", line.strip())  # Check if the line ends with '+n'
    if match:
        n = int(match.group(1))
        total_sum += n

percentage = (count / total_sum) * 100
print(
    f"{percentage:.2f}% of points(+4 and above decays) excluded due to difficulties in parsing."
)

with open("final_result.txt", "r") as f:
    lines = f.readlines()

with open("final_result.txt", "w") as f:
    skip_next = 0
    for line in lines:
        if skip_next > 0:
            skip_next -= 1
            continue
        match = re.search(r"\+(\d+)$", line.strip())  # Check if the line ends with '+n'
        if match:
            n = int(match.group(1))
            if n > 3:  # Ensure n is greater than 2
                skip_next = n
                continue
        f.write(line)


# open final_result.txt and if any line ends with +2, duplicate that line below it
with open("final_result.txt", "r") as f:
    lines = f.readlines()

with open("final_result.txt", "w") as f:
    for line in lines:
        f.write(line)
        if line.strip().endswith("+2"):
            f.write(line)
# print number of lines in final_result.txt
with open("final_result.txt", "r") as f:
    lines = f.readlines()
    print(len(lines))
# open final_result.txt, if any line repeats more than once, swap its position with the line below it


def swap_repeated_lines(file_path):
    # Read the contents of the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Iterate over the lines
    prev_line = None
    for i in range(len(lines)):
        if prev_line == lines[i]:
            # Swap current line with the line below it
            lines[i], lines[i + 1] = lines[i + 1], lines[i]
            prev_line = None  # Reset prev_line after swapping
        else:
            prev_line = lines[i]

    # Write the modified lines back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)


# Usage example
file_path = "final_result.txt"
swap_repeated_lines(file_path)
# print number of lines in final_result.txt
with open("final_result.txt", "r") as f:
    lines = f.readlines()
    print(len(lines))


# Duplicate lines ending with +3 two times below them
with open("final_result.txt", "r") as f:
    lines = f.readlines()

with open("final_result_modified.txt", "w") as f:
    for line in lines:
        f.write(line)
        if line.strip().endswith("+3"):
            f.write(line)
            f.write(line)

# Swap lines i+1 and i+4 if line i ends with +3 and start scanning from i+6
with open("final_result_modified.txt", "r") as f:
    lines = f.readlines()

with open("final_result_final.txt", "w") as f:
    skip_next = 0
    for i in range(len(lines)):
        if skip_next > 0:
            skip_next -= 1
            continue
        f.write(lines[i])
        if lines[i].strip().endswith("+3"):
            if i + 4 < len(lines):  # Make sure there are enough lines to swap
                lines[i + 1], lines[i + 4] = lines[i + 4], lines[i + 1]
                skip_next = 5
            else:
                f.write("Error: Not enough lines to perform swap.\n")


with open("final_result.txt", "r") as file:
    lines = file.readlines()

combined_lines = []
for i in range(0, len(lines), 2):
    combined_lines.append(lines[i].strip() + "," + lines[i + 1].strip())

with open("final_result.txt", "w") as file:
    file.write("\n".join(combined_lines))

# print number of lines in final_result.txt
with open("final_result.txt", "r") as f:
    lines = f.readlines()
    print(len(lines))
    
######---DONE PARSING----#####
    
########---PANDS----##########
    
    
# open final_result.txt and load into a pandas dataframe
import pandas as pd

df = pd.read_csv("final_result.txt", header=None)

# name the coloumns: id  , cellId0 ,cellId1  , energy      ,   position (x,y,z)        ,  nMCParticles MC contribution: prim. PDG,   energy_part  ,   time,    length  , sec. PDG and stepPosition (x,y,z)
df.columns = [
    "id",
    "cellId0",
    "cellId1",
    "energy",
    "position x",
    "position y",
    "position z",
    "nMCParticles",
    "mc contri prim. PDG",
    "energy_part",
    "time",
    "length",
    "sec_PDG",
    "stepPosition x",
    "stepPosition y",
    "stepPosition z",
]

# convert all the columns to the numeric type
df = df.apply(pd.to_numeric, errors="coerce")



#########---PLOT TEST----#########


# plot a histogram of the energy column
import matplotlib.pyplot as plt

df["energy"].plot(kind="hist", bins=20)
# log scale
plt.yscale("log")

plt.show()
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming df is your DataFrame containing x, y, z, and energy columns

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Extract data
x = df["position x"]
y = df["position y"]
z = df["position z"]
c = df["energy"]

# Scale energy values for better visualization
c = c.apply(lambda x: 0 if x == 0 else -np.log10(x))

# Use Seaborn for enhanced styling
sns.set(style="darkgrid")

# Create scatter plot with color mapping
img = ax.scatter(x, y, z, c=c, cmap="rainbow")

# Add color bar for energy levels
cbar = fig.colorbar(img)
cbar.set_label("Energy (log scale)")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("ECalBarrelCollection Position vs Energy (Event ID:000/999)")

# Set dark background
ax.patch.set_facecolor("black")

# Adjust perspective and viewing angle
ax.view_init(elev=20, azim=120)

# Show plot
plt.show()
# repeat for position x, y, z vs time
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Extract data
x = df["position x"]
y = df["position y"]
z = df["position z"]
c = df["time"]


# Use Seaborn for enhanced styling
sns.set(style="darkgrid")

# Create scatter plot with color mapping
img = ax.scatter(x, y, z, c=c, cmap="rainbow")
# colorbar values should be from 0 to max value of time
img.set_clim(df["time"].min(), df["time"].max())
# print all the negative values of time


# Add color bar for time levels
cbar = fig.colorbar(img)
cbar.set_label("Time")
# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("ECalBarrelCollection Position vs Time (Event ID:000/999)")
# Set dark background
ax.patch.set_facecolor("black")

# Adjust perspective and viewing angle
ax.view_init(elev=20, azim=120)

# Show plot
plt.show()
