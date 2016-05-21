import argparse

parser = argparse.ArgumentParser(description="Removes all lines starting with a #-tag.")
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (CoNLL format).", required=True)
args = parser.parse_args()

outfile = open(args.outfile, 'w+')
for line in open(args.infile, 'r'):
    if line.strip():
        first = line.split('\t')[0]
        if not '-' in first:
            print(line.strip(), file=outfile)
    else:
        print(line.strip(), file=outfile)

outfile.close()
