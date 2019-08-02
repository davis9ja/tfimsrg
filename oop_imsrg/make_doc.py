import os
import sys
import glob

doc = ""
mv = ""
cmd = ""
src = ""
main_prefix = ""
# print(os.name)
if os.name == 'posix':
    src = './oop_imsrg/'
    doc_dir = src+"doc/"
    mv = "mv"
    main_prefix = "./"
    # cmd = "{:s} *.html {:s}".format(mv, doc_dir)
elif os.name == 'nt':
    src = '.\\oop_imsrg\\'
    doc_dir = src+"doc\\"
    mv = "move"
    main_prefix = ".\\"
    # cmd = "{:s} -Path .\\*.html -Destination .\\{:s}".format(mv, doc_dir)
else:
    print("Could not detect operating system")
    exit()

if not os.path.exists(doc_dir):
    os.mkdir(doc_dir)

os.system('pydoc -w '+src+'hamiltonian.py')
os.system('pydoc -w '+src+'flow.py')
os.system('pydoc -w '+src+'generator.py')
os.system('pydoc -w '+main_prefix+'main.py')
os.system('pydoc -w '+src+'occupation_tensors.py')
os.system('pydoc -w '+src+'plot_data.py')
os.system("{:s} *.html {:s}".format(mv, doc_dir))
