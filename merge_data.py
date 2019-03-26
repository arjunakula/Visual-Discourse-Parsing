import sys,os
import glob
import re
import numpy as np
import random

mapping_file_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/all_data/"
train_file_name = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/alldata.train"
dev_file_name = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/alldata.dev"
test_file_name = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/alldata.test"

ftr = open(train_file_name,"a")
fdv = open(dev_file_name,"a")
fts = open(test_file_name,"a")

subdirfiles1 = glob.glob(mapping_file_extension+"*.train")
subdirfiles2 = glob.glob(mapping_file_extension+"*.dev")
subdirfiles3 = glob.glob(mapping_file_extension+"*.test")

for i in subdirfiles1:
        fp = open(i,"r")
        data = fp.read().lower().replace('<cause> <','<vc> <').replace('cause <','<vc> <').replace('vc <','<vc> <').replace('nvr <','<nvr> <').replace('vr <','<vr> <').replace('evaluation <','<evaluation> <').replace('contrast <','<contrast> <').replace('interpretation <','<interpretation> <')
        ftr.write(data)
        fp.close()
ftr.close()

for i in subdirfiles2:
        fp = open(i,"r")
        data = fp.read().lower().replace('<cause> <','<vc> <').replace('cause <','<vc> <').replace('vc <','<vc> <').replace('nvr <','<nvr> <').replace('vr <','<vr> <').replace('evaluation <','<evaluation> <').replace('contrast <','<contrast> <').replace('interpretation <','<interpretation> <')
        fdv.write(data)
        fp.close()
fdv.close()

for i in subdirfiles3:
        fp = open(i,"r")
        data = fp.read().lower().replace('<cause> <','<vc> <').replace('cause <','<vc> <').replace('vc <','<vc> <').replace('nvr <','<nvr> <').replace('vr <','<vr> <').replace('evaluation <','<evaluation> <').replace('contrast <','<contrast> <').replace('interpretation <','<interpretation> <')
        fts.write(data)
        fp.close()
fts.close()
