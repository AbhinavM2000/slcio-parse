import re

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



#open final_result.txt and if any line ends with +2, duplicate that line below it
with open('final_result.txt', 'r') as f:
    lines = f.readlines()

with open('final_result.txt', 'w') as f:
    for line in lines:
        f.write(line)
        if line.strip().endswith('+2'):
            f.write(line)

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
