import os


def findFileWithKeyword(root_path, keyword):
    """
    Returns the path of the first file found with the given keyword.
    Explicitly discards files beginning with a dot, e.g. '._abs'.
    """
    for root, dirs, files in os.walk(os.path.join(root_path)):
        for _file in files:
            if (_file.find(keyword) > -1) and (not _file.startswith(".")):
                found_file = os.path.join(root, _file)
                return found_file
