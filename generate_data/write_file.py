
#  class WriteFile:
def write_secondary(filename = "", dots = "", protein_sequence = "", brackets = "", rna_seq = ""):
    ps = protein_sequence.upper()
    rnaseq = rna_seq.lower()

    the_string = dots + brackets + "\n" + ps + rnaseq

    with open(filename,"w") as outfile:
        outfile.write(the_string)
        outfile.close()


def write_fasta(filename = "", header = "", last_chain = "", protein_sequence = "", rna_seq = ""):
    ps = protein_sequence.upper()
    rnaseq = rna_seq.lower()
            
    with open(filename,"w") as outfile:
        outfile.write("{}{}\n{}{}".format(header,last_chain,ps,rnaseq))
        outfile.close()


def write_data(filename = "", data =[]):
    with open(filename,"w") as outfile:
        outfile.writelines(data)
        outfile.close()