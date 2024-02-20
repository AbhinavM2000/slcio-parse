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
