"""
    Source file that contains the misc utility or wraper functions
"""

import os 


"""
    return true if the path contains a file or a directory
    @params path <string> directory or file path
    @returns <boolean> returns true if either is a file or directory
"""
def exists(path):
    return (os.path.isfile(path) or os.path.isdir(path)) 

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def delete(path):
    if exists(path):
        try:
            os.remove(path)
        except:
            os.rmdir(path)
    else:
        print("File/Folder {} doesnt exist".format(path))