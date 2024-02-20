#code to parse the text
#->                          +2|+2.738e-04|+9.607e+00|+0.000e+00| (+0, +0.000e+00, +0.000e+00, +0.000e+00)
import re
def arrow_line_parse(line):
    if (line[0:2] =='->'):
        line=line[2:].strip()
        line = line.split('|')
        return line
#code to parse the text
#[00000034] |20448660|-15269951|+2.738e-04|+1.632e+03, +5.714e+02, -1.193e+03|          +1
def header_line_parse(input_str):
    # Regular expression pattern to match the desired values
    pattern = r'\[(\d+)\]\s*\|([-+]?\d+)\|\s*([-+]?\d+)\|\s*([-+]?\d+\.\d+e[-+]?\d+)\|\s*([-+]?\d+\.\d+e[-+]?\d+)\s*,\s*([-+]?\d+\.\d+e[-+]?\d+)\s*,\s*([-+]?\d+\.\d+e[-+]?\d+)\|\s*([-+]?\d+)'

    # Using re.search to find the first match
    match = re.search(pattern, input_str)

    # Extracting the values from the match
    if match:
        return list(match.groups())
    else:
        return []
