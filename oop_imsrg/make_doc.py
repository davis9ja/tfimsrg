import os
import sys
import glob

doc = ""
mv = ""
cmd = ""
# print(os.name)
if os.name == 'posix':
    doc_dir = "doc/"
    mv = "mv"
    # cmd = "{:s} *.html {:s}".format(mv, doc_dir)
elif os.name == 'nt':
    doc_dir = "doc\\"
    mv = "move"
    # cmd = "{:s} -Path .\\*.html -Destination .\\{:s}".format(mv, doc_dir)
else:
    print("Could not detect operating system")
    exit()

if not os.path.exists(doc_dir):
    os.mkdir(doc_dir)

os.system('pydoc -w hamiltonian flow generator main occupation_tensors plot_data')
os.system("{:s} *.html {:s}".format(mv, doc_dir))
