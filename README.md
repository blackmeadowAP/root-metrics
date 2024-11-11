# root-metrics
Sripts for recognising roots on arabidopsis thaliana dishes and measure their length

This software was created for analyzing petri dishes of arabidopsis thaliana in Petri dishes for measuring changes in roots lengths in stress conditions.

petri dishes cutter - finds ROI(region of interest) and cuts it from raw photos
mask extractor - mask r-cnn for extracting root masks from ROI images
root metrics - counts lenghts of roots in masked images
