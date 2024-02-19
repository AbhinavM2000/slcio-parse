#code to parse the text
#->                          +2|+2.738e-04|+9.607e+00|+0.000e+00| (+0, +0.000e+00, +0.000e+00, +0.000e+00)

def arrow_line_parse(line):
    if (line[0:2] =='->'):
        line=line[2:].strip()
        line = line.split('|')
        return line


    
