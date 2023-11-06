from pathlib import Path
DEFAULT_ABIDE_NAME = "__ABIDE_dataset"
root_folder = Path(__file__).parent.parent.absolute()
DEFAULT_ABIDE_ROOT_LOCATION = root_folder/DEFAULT_ABIDE_NAME
ABIDE_PCP = 'ABIDE_pcp'
PIPELINE = 'cpac'
STRATEGY = 'filt_noglobal'
ATLAS_NAME = 'ho'

DEFAULT_ABIDE_LOCATION = DEFAULT_ABIDE_ROOT_LOCATION/ ABIDE_PCP / PIPELINE/ STRATEGY
