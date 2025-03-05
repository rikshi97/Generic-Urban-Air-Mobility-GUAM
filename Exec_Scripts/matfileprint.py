import h5py

# Load the .mat file
with h5py.File('exam_PW_Bezier_Traj.mat', 'r') as mat_file:

# Access the variables in the .mat file
    print(mat_file)