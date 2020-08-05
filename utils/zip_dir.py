import os


def zipdir(path_, ziph):
    for root, dirs, files in os.walk(path_, topdown=True):
        dirs[:] = [d for d in dirs if d not in {"__pycache__"}]
        for file_ in files:
            if os.path.splitext(file_)[1] == ".pyc":
                continue
            ziph.write(os.path.join(root, file_),
                       arcname=os.path.relpath(os.path.join(root, file_), os.path.join(path_, '..')))
