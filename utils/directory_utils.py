import shutil
from pathlib import Path
import json
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def delete_folder(folder_name):
    try:
        shutil.rmtree(folder_name)
        print(f"Successfully deleted the '{folder_name}' folder and its contents.")
    except OSError as e:
        print(f"Error: {e} - '{folder_name}' folder was not deleted.")
