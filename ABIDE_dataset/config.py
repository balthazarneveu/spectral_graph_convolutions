from pathlib import Path
DEFAULT_ABIDE_NAME = "__ABIDE_dataset"
root_folder = Path(__file__).parent.parent.absolute()
DEFAULT_ABIDE_ROOT_LOCATION = root_folder/DEFAULT_ABIDE_NAME

pipeline = 'cpac'
strategy = 'filt_noglobal'

DEFAULT_ABIDE_LOCATION = DEFAULT_ABIDE_ROOT_LOCATION/ 'ABIDE_pcp'/ pipeline/ strategy
