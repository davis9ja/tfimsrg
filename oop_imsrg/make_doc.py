import os
import sys

doc = ""

if os.name == 'posix':
	doc_dir = "doc/"
elif os.name == 'nt':
	doc_dir = "doc\\"
else:
	print("Could not detect operating system")
	exit()

if not os.path.exists(doc_dir):
	os.mkdir(doc_dir)

os.system('pydoc -w *.py')
os.system('mv *.html {:s}'.format(doc_dir))