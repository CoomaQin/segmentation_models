def get_filename_from_path(path):
    file = path.split("/")[-1]
    filename = file.split(".")[0]
    return filename
