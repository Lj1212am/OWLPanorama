# OWLPanorama

This project aims at creating a panorama video out of 4 separate GoPro videos. Which takes the 4 videos in mp4 as input and generate a panorama video.

# Hardware 
Four GoPro 10 hero blacks
Desktop with 13th generation Intel i7 and 32GB RAM. Note: Long panorama videos  could be ram consuming, if you come into a ram crash, consider to retry with short clips. 

# Running 
python cvPanoroma.py

Update the file paths to where you have the four videos saved.

# Tuning 
The videos we use are taken by a GoPro 10 series in 1080p with wide lens, which has 118 degrees FOV horizontally and 69 degrees FOV vertically, if different lens settings are used, recalculate the parameters in Barrelbackprojection.py

# Contact us
zruijun@seas.upenn.edu
lmilburn@seas.upenn.edu
