import os
dirname = "/home/ouni/CODE/GelSight/markered"
files = sorted(os.listdir(dirname))
Nfiles = len(files)
os.chdir(dirname)
for i,file in enumerate(files):
    os.rename(file, "%04d%s"%(i,os.path.splitext(file)[1]))