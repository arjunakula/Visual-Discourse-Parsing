import sys,os
import glob
import re
import numpy as np
import random

mapping_file_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/frame_rst_mappings/"
train_file_name = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/frame_rst_mappings/data_tmp.train"
dev_file_name = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/frame_rst_mappings/data_tmp.dev"
test_file_name = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/frame_rst_mappings/data_tmp.test"


#total_samples = 
#random.shuffle(total_samples)
subdirfiles = glob.glob(mapping_file_extension+"*.txt")
total_samples = len(subdirfiles)
all_samples = range(0,len(subdirfiles))

train_size = (total_samples*78)/100
dev_size = (total_samples*10)/100

train_samples = all_samples[0:train_size-1]
dev_samples = all_samples[train_size-1:train_size-1+dev_size]
test_samples = all_samples[train_size-1+dev_size:total_samples-1]

ftr = open(train_file_name,"a")
fdv = open(dev_file_name,"a")
fts = open(test_file_name,"a")

for i in train_samples:
	fp = open(subdirfiles[i],"r")
	data = fp.read().split("\n")
	for j in range(0,len(data)-1):
		ftr.write(data[j].replace('[','').replace(']','').replace(',','')+"\n")
	
	ftr.write(data[len(data)-1]+"\n")
	ftr.write("\n")
	fp.close()
ftr.close()


for i in dev_samples:
	fp = open(subdirfiles[i],"r")
	data = fp.read().split("\n")
	for j in range(0,len(data)-1):
		fdv.write(data[j].replace('[','').replace(']','').replace(',','')+"\n")
	
	fdv.write(data[len(data)-1]+"\n")
	fdv.write("\n")
	fp.close()
fdv.close()


for i in test_samples:
	fp = open(subdirfiles[i],"r")
	data = fp.read().split("\n")
	for j in range(0,len(data)-1):
		fts.write(data[j].replace('[','').replace(']','').replace(',','')+"\n")
	
	fts.write(data[len(data)-1]+"\n")
	fts.write("\n")
	fp.close()

fts.close()
