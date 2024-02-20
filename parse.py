import pandas as pd
import re

def arrow_parse(line):
    if line.startswith('->') and re.search(r'[-+]\d', line):
        line = line[2:].strip()
        line = line.split('|')
        return line
    else:
        print("Error: Invalid arrow line format:", line)

def header_parse(input_str):
    # Regular expression pattern to match the desired values
    pattern = r'\[(\d+)\]\s*\|([-+]?\d+)\|\s*([-+]?\d+)\|\s*([-+]?\d+\.\d+e[-+]?\d+)\|\s*([-+]?\d+\.\d+e[-+]?\d+)\s*,\s*([-+]?\d+\.\d+e[-+]?\d+)\s*,\s*([-+]?\d+\.\d+e[-+]?\d+)\|\s*([-+]?\d+)'

    # Using re.search to find the first match
    match = re.search(pattern, input_str)

    # Extracting the values from the match
    if match:
        return list(match.groups())
    else:
        print("Error: Invalid header line format:", input_str)
        return []

def read_and_process_file(file_path):
    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['list_header', 'list_arrow'])

    with open(file_path, 'r') as file:
        lines = file.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            print("Processing line:", line)
            if line.startswith('['):
                header_list = header_parse(line)
                if header_list:
                    print("Header parsed:", header_list)
                    idx += 1
                    next_line = lines[idx].strip()
                    print("Next line:", next_line)
                    if line.endswith('+2'):
                        print("Encountered +2 arrow lines.")
                        arrow_lists = []
                        for _ in range(2):
                            
                            arrow_line = lines[idx].strip()
                            print("Processing arrow line:", arrow_line)
                            arrow_list = arrow_parse(arrow_line)
                            if arrow_list:
                                arrow_lists.append(arrow_list)
                            idx += 1
                        if arrow_lists:
                            print("Arrow lists parsed:", arrow_lists)
                            df = df.append({'list_header': header_list, 'list_arrow': arrow_lists}, ignore_index=True)
                    elif line.endswith('+1'):
                        print("Encountered +1 arrow line.")
                        idx += 1
                        arrow_line = lines[idx].strip()
                        print("Processing arrow line:", arrow_line)
                        arrow_list = arrow_parse(arrow_line)
                        if arrow_list:
                            print("Arrow list parsed:", arrow_list)
                            df = df.append({'list_header': header_list, 'list_arrow': [arrow_list]}, ignore_index=True)
                    else:
                        print("Error: Invalid arrow line specifier:", next_line)
            idx += 1
    return df

# Test the function with a file
file_path = 'data.txt'  # Provide the path to your text file
df = read_and_process_file(file_path)
print("Final DataFrame:")
print(df)
