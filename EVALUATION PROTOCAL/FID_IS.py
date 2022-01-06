
pip install torch-fidelity

# IS
fidelity --gpu 0 --isc --input1 img_dir1/

#FID
fidelity --gpu 0 --fid --input1 img_dir1/ --input2 img_dir2/

#KID
fidelity --gpu 0 --kid --input1 img_dir1/ --input2 img_dir2/

