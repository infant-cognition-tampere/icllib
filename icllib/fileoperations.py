# Find all files on subdirectories of parentdir and transfer those matching the
# regular expression to destinationdir.
#
# Usage:
#
# import fileoperations
# fileoperations.collectfiles("my_parentdir", "my_destdir",".txt")




import os
import shutil
import re

def collectfiles(parentdir, destinationdir, pattern):
    filelist = os.listdir(parentdir)

    movefilecount = 0

    for f in filelist:
        filepath = os.path.join(parentdir, f)
        if os.path.isdir(filepath):
            collectfiles(filepath, destinationdir, pattern)
        else:
            match = re.search(pattern, f)
            if match is not None:
                shutil.move(filepath, destinationdir)
                print("Moved " + filepath + " to " + os.path.join(destinationdir, f))
                movefilecount = movefilecount + 1

    print("Total of " + str(movefilecount) + " files transferred.")