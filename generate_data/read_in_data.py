# read methods
def read_in(filename = ""):
    with open(filename,"r") as infile:
        lines = infile.readlines()
        infile.close()
    return(lines)