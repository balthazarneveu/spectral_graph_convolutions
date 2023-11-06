# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
# This code is written from the github : https://github.com/parisots/population-gcn/tree/master of the studied article

import argparse
from nilearn import datasets
import os
import shutil
import numpy as np
import ABIDE_graph as Reader


# Get the list of subject IDs
def get_ids(data_folder, num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

def fetch_filenames(data_folder, subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)  # change the current/working directory to the directory where the data are stored
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Download preprocessed datain chosen directory')
    parser.add_argument('-o', '--out_dir', required=True, type=str, help='Path to local folder to download files to')
    parser.add_argument('-n', '--nb_subjects', type=int, default=871, help='Number of data/subject to download')
    args = parser.parse_args()

    # Selected pipeline
    pipeline = 'cpac'
    strategy = 'filt_noglobal'

    # Input data variables
    num_subjects = args.nb_subjects # Number of subjects
    root_folder =  args.out_dir #'/path/to/data/'
    data_folder = os.path.join(root_folder, 'ABIDE_pcp', pipeline, strategy)

    # Files to fetch
    files = ['rois_ho']

    filemapping = {'func_preproc': 'func_preproc.nii.gz', # to you use for fMRI images
                'rois_ho': 'rois_ho.1D'}

    if not os.path.exists(data_folder): os.makedirs(data_folder)
    shutil.copyfile('./subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt'))

    # Download database files
    abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
                                    band_pass_filtering=True, global_signal_regression=False, derivatives=files)


    subject_IDs = get_ids(data_folder=data_folder, num_subjects=num_subjects)
    subject_IDs = subject_IDs.tolist()

    # Create a folder for each subject
    for s, fname in zip(subject_IDs, fetch_filenames(data_folder=data_folder, subject_IDs=subject_IDs, file_type=files[0])):
        subject_folder = os.path.join(data_folder, s)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                shutil.move(base + filemapping[fl], subject_folder)

    # Compute and save connectivity matrices
    time_series = Reader.get_timeseries(data_folder=data_folder, subject_list=subject_IDs, atlas_name='ho')
    for i in range(len(subject_IDs)):
            Reader.compute_subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation', save_path=data_folder)

