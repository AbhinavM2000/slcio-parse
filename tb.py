import os
import subprocess

def extract_data(input_file):
    lines = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    start_index = 23  # Line number 24 (0-indexed)
    end_index = None
    for i in range(start_index, len(lines)):
        if lines[i].startswith('------------'):
            end_index = i
            break
    
    if end_index is not None:
        processed_lines = []
        for line in lines[start_index:end_index]:
            line = line.lstrip()  # Remove leading whitespaces
            if line and not line.startswith('id-fields'):  # Ignore lines starting with 'id-fields'
                processed_lines.append(line)
        
        with open('temp.txt', 'w') as f:
            f.writelines(processed_lines)
    else:
        print("End marker '----' not found in file:", input_file)

def process_file(input_file, angle):
    extract_data(input_file)
    subprocess.run(['python3', 'parse.py', 'temp.txt', 'final_result.txt', 'time', '--angle', str(angle), '--save_image'])
def main():
    #current directory
    directory = '.'
    angle = 0  # Start angle
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('event'):
            input_file = os.path.join(directory, filename)
            process_file(input_file, angle)
            angle += 0.18  # Increment angle by 2 degrees

if __name__ == "__main__":
    main()
