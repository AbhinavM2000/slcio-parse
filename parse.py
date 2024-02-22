import re
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go



def process_line(line):
    # Replace special characters with ','
    line = re.sub(r'[^\w\s.,+-]', ',', line)
    # Remove unnecessary spaces
    line = ' '.join(line.split())
    return line

def remove_extra_commas(lines):
    cleaned_lines = []
    for line in lines:
        if line[:2] == '-,':
            line = line[2:]
        parts = line.split(',')
        cleaned_parts = [part.strip() for part in parts if part.strip()]
        cleaned_line = ','.join(cleaned_parts)
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = [process_line(line) for line in lines]
    cleaned_lines = remove_extra_commas(processed_lines)

    with open(output_file, 'w') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

# Replace 'data.txt' and 'final_result.txt' with your input and output file paths
process_file('data.txt', 'final_result.txt')

#print number of lines in final_result.txt
with open('final_result.txt', 'r') as f:
    lines = f.readlines()
    print(len(lines))

import re
#open final_result.txt and if any line ends with +n, where n greater than 3, calculate sum of all such n = count. Then print % = count / (sum of all n including those less than 3)
with open('final_result.txt', 'r') as f:
    lines = f.readlines()

sum_n = 0
count = 0
for line in lines:
    match = re.search(r'\+(\d+)$', line.strip()) # Check if the line ends with '+n'
    if match:
        n = int(match.group(1))
        if n > 3:  # Ensure n is greater than 3
            sum_n += n
            count += 1

total_sum = sum_n
for line in lines:
    match = re.search(r'\+(\d+)$', line.strip()) # Check if the line ends with '+n'
    if match:
        n = int(match.group(1))
        total_sum += n

percentage = (count / total_sum) * 100
print(f"{percentage:.2f}% of points(+4 and above decays) excluded due to difficulties in parsing.")

with open('final_result.txt', 'r') as f:
    lines = f.readlines()

with open('final_result.txt', 'w') as f:
    skip_next = 0
    for line in lines:
        if skip_next > 0:
            skip_next -= 1
            continue
        match = re.search(r'\+(\d+)$', line.strip()) # Check if the line ends with '+n'
        if match:
            n = int(match.group(1))
            if n > 3:  # Ensure n is greater than 2
                skip_next = n
                continue
        f.write(line)




#open final_result.txt and if any line ends with +2, duplicate that line below it
with open('final_result.txt', 'r') as f:
    lines = f.readlines()

with open('final_result.txt', 'w') as f:
    for line in lines:
        f.write(line)
        if line.strip().endswith('+2'):
            f.write(line)
#print number of lines in final_result.txt
with open('final_result.txt', 'r') as f:
    lines = f.readlines()
    print(len(lines))
#open final_result.txt, if any line repeats more than once, swap its position with the line below it

def swap_repeated_lines(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate over the lines
    prev_line = None
    for i in range(len(lines)):
        if prev_line == lines[i]:
            # Swap current line with the line below it
            lines[i], lines[i+1] = lines[i+1], lines[i]
            prev_line = None  # Reset prev_line after swapping
        else:
            prev_line = lines[i]

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Usage example
file_path = 'final_result.txt'
swap_repeated_lines(file_path)
#print number of lines in final_result.txt
with open('final_result.txt', 'r') as f:
    lines = f.readlines()
    print(len(lines))


# Duplicate lines ending with +3 two times below them
with open('final_result.txt', 'r') as f:
    lines = f.readlines()

with open('final_result_modified.txt', 'w') as f:
    for line in lines:
        f.write(line)
        if line.strip().endswith('+3'):
            f.write(line)
            f.write(line)

# Swap lines i+1 and i+4 if line i ends with +3 and start scanning from i+6
with open('final_result_modified.txt', 'r') as f:
    lines = f.readlines()

with open('final_result_final.txt', 'w') as f:
    skip_next = 0
    for i in range(len(lines)):
        if skip_next > 0:
            skip_next -= 1
            continue
        f.write(lines[i])
        if lines[i].strip().endswith('+3'):
            if i + 4 < len(lines):  # Make sure there are enough lines to swap
                lines[i+1], lines[i+4] = lines[i+4], lines[i+1]
                skip_next = 5
            else:
                f.write("Error: Not enough lines to perform swap.\n")





with open("final_result.txt", "r") as file:
    lines = file.readlines()

combined_lines = []
for i in range(0, len(lines), 2):
    combined_lines.append(lines[i].strip() + "," + lines[i+1].strip())

with open("final_result.txt", "w") as file:
    file.write("\n".join(combined_lines))

#print number of lines in final_result.txt
with open('final_result.txt', 'r') as f:
    lines = f.readlines()
    print(len(lines))
#open final_result.txt and load into a pandas dataframe
import pandas as pd

df = pd.read_csv('final_result.txt', header=None)

 # name the coloumns: id  , cellId0 ,cellId1  , energy      ,   position (x,y,z)        ,  nMCParticles MC contribution: prim. PDG,   energy_part  ,   time,    length  , sec. PDG and stepPosition (x,y,z)
df.columns = ['id', 'cellId0', 'cellId1', 'energy', 'position x','position y','position z', 'nMCParticles', 'mc contri prim. PDG', 'energy_part', 'time', 'length', 'sec_PDG', 'stepPosition x', 'stepPosition y', 'stepPosition z']

#convert all the columns to the numeric type
df = df.apply(pd.to_numeric, errors='coerce')
#plot a histogram of the energy column
import plotly.graph_objs as go

# Create histogram trace
import plotly.graph_objects as go

# Create histogram trace
histogram = go.Histogram(
    x=df['energy'],
    nbinsx=20,
    marker=dict(color='royalblue')
)

# Create layout
layout = go.Layout(
    title='Energy Distribution ECalBarrelCollection (Event ID:000/999)',
    xaxis=dict(title='Energy', color='white'),
    yaxis=dict(title='Count (log scale)', type='log', color='white'),
    bargap=0.05,
    bargroupgap=0.1,
    plot_bgcolor='rgb(51, 51, 51)',
    paper_bgcolor='rgb(51, 51, 51)',
    font=dict(color='white'),
    autosize=True,
)

# Create figure
fig = go.Figure(data=[histogram], layout=layout)

# Update color for dark theme
fig.update_layout(
    xaxis=dict(linecolor='white'),
    yaxis=dict(linecolor='white'),
    bargap=0.05,
    bargroupgap=0.1,
    plot_bgcolor='rgb(51, 51, 51)',
    paper_bgcolor='rgb(51, 51, 51)',
    font=dict(color='white')
)

# Show plot
#fig.show()



import plotly.graph_objs as go

# Scale energy values for better visualization
df['scaled_energy'] = df['energy'].apply(lambda x: 0 if x == 0 else -np.log10(x))

# Create scatter plot trace
scatter = go.Scatter3d(
    x=df['position x'],
    y=df['position y'],
    z=df['position z'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['scaled_energy'],
        colorscale='Rainbow',
        colorbar=dict(title='Energy (log scale)'),
        line=dict(color='white', width=0),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='ECalBarrelCollection Position vs Energy (Event ID:000/999)',
    scene=dict(
        xaxis=dict(showbackground=False, showline=True, linecolor='white', title='X', showgrid=False, showticklabels=True),
        yaxis=dict(showbackground=False, showline=True, linecolor='white', title='Y', showgrid=False, showticklabels=True),
        zaxis=dict(showbackground=False, showline=True, linecolor='white', title='Z', showgrid=False, showticklabels=True),
        bgcolor='black'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor='black',
    font=dict(color='white')
)

# Create figure
fig = go.Figure(data=[scatter], layout=layout)

# Show plot
#fig.show()


# For the second part:

# Create scatter plot trace
scatter = go.Scatter3d(
    x=df['position x'],
    y=df['position y'],
    z=df['position z'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['time'],
        colorscale='Rainbow',
        colorbar=dict(title='Time'),
        line=dict(color='white', width=0),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='ECalBarrelCollection Position vs Time (Event ID:000/999)',
    scene=dict(
        xaxis=dict(showbackground=False, showline=True, linecolor='white', title='X', showgrid=False, showticklabels=True),
        yaxis=dict(showbackground=False, showline=True, linecolor='white', title='Y', showgrid=False, showticklabels=True),
        zaxis=dict(showbackground=False, showline=True, linecolor='white', title='Z', showgrid=False, showticklabels=True),
        bgcolor='black'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor='black',
    font=dict(color='white')
)

# Create figure
fig = go.Figure(data=[scatter], layout=layout)

# Show plot
fig.show()




###---UNROLL CYLINDER INTO RECTANGLE (TESTING)---###

import numpy as np
import plotly.graph_objs as go

# Assuming you have already loaded your data into a DataFrame called df
# Extract data
x = df['position x']
y = df['position y']
z = df['position z']
c = df['time']

# Convert cylindrical coordinates to rectangular coordinates


#print points with lowest z coordinate value
min_z = df[df['position z'] == df['position z'].min()]
O=[0,0,min_z['position z'].values[0]]
A=[min_z['position x'].values[0],min_z['position y'].values[0],min_z['position z'].values[0]]
#print O and A
print(f"O: {O}")
print(f"A: {A}")
#find angle between OA and every other point, check cross product of OA and OB, if positive angle = angle otherwise angle = -angle
def angle_between_points(O, A, B):
    # Calculate vectors OA and OB
    OA = np.array(A) - np.array(O)
    OB = np.array(B) - np.array(O)
    # Calculate angle between OA and OB
    angle = np.arccos(np.dot(OA, OB) / (np.linalg.norm(OA) * np.linalg.norm(OB)))
    # Calculate cross product of OA and OB
    cross_product = np.cross(OA, OB)
    #radius ob norm
    # If cross product is positive, angle is positive, otherwise negative
    if cross_product[2] > 0:
        return angle
    else:
        return -angle
# iterate over all the points and calculate the angle between OA and OB, if B=0, then skip
angles = []
zvalues = []
for i in range(len(df)):
    B = [df['position x'].iloc[i], df['position y'].iloc[i], -2203.0]
    if (B==O):
        print(f"O: {O}")
        continue
    angles.append(angle_between_points(O, A, B))
    zvalues.append(df['position z'].iloc[i])
# For the z-coordinate, it remains the same as the z of the cylinder
z_rect = z
# Create scatter plot trace
scatter = go.Scatter(
    x=angles,
    y=z_rect,
    mode='markers',
    marker=dict(
        size=8,
        color=c,
        colorscale='Rainbow',
        colorbar=dict(title='Time'),
        opacity=0.8
    )
)
# Create layout
layout = go.Layout(
    title='ECalBarrelCollection Position vs Time (Event ID:000/999)',
    xaxis=dict(title='Theta (degrees)'),
    yaxis=dict(title='Z'),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white')
)

# Create figure
fig = go.Figure(data=[scatter], layout=layout)

# Show plot
fig.show()



# Scale energy values for better visualization
df['scaled_energy'] = df['energy'].apply(lambda x: 0 if x == 0 else -np.log10(x))

# Create scatter plot trace
scatter = go.Scatter3d(
    x=df['position x'],
    y=df['position y'],
    z=df['position z'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['scaled_energy'],
        colorscale='Rainbow',
        colorbar=dict(title='Energy (log scale)'),
        line=dict(color='white', width=0),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='ECalBarrelCollection Position vs Energy (Event ID:000/999)',
    scene=dict(
        xaxis=dict(showbackground=False, showline=True, linecolor='white', title='X', showgrid=False, showticklabels=True),
        yaxis=dict(showbackground=False, showline=True, linecolor='white', title='Y', showgrid=False, showticklabels=True),
        zaxis=dict(showbackground=False, showline=True, linecolor='white', title='Z', showgrid=False, showticklabels=True),
        bgcolor='black'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor='black',
    font=dict(color='white')
)

# Create figure
fig = go.Figure(data=[scatter], layout=layout)

# Show plot
#fig.show()


# calclate radius
r = np.sqrt(df['position x']**2 + df['position y']**2)

# Create scatter plot trace
scatter = go.Scatter3d(
    x=np.cos(angles)*r[0],
    y=np.sin(angles)*r[0],
    z=df['position z'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['time'],
        colorscale='Rainbow',
        colorbar=dict(title='Time'),
        line=dict(color='white', width=0),
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='ECalBarrelCollection Position vs Time (Event ID:000/999)',
    scene=dict(
        xaxis=dict(showbackground=False, showline=True, linecolor='white', title='X', showgrid=False, showticklabels=True),
        yaxis=dict(showbackground=False, showline=True, linecolor='white', title='Y', showgrid=False, showticklabels=True),
        zaxis=dict(showbackground=False, showline=True, linecolor='white', title='Z', showgrid=False, showticklabels=True),
        bgcolor='black'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    paper_bgcolor='black',
    font=dict(color='white')
)

# Create figure
fig = go.Figure(data=[scatter], layout=layout)

# Show plot
fig.show()




