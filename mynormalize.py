import sys,os
import glob
import re
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

mapping_file_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/data_to_normalize/"

subdirfiles = glob.glob(mapping_file_extension+"*")
total_samples = len(subdirfiles)

for i in subdirfiles:
        fw = open(i+"new","a")
        fr = open(i,"r")
        data = fr.read().split("\n\n")
	print len(data)
        for j in range(0,len(data)):
		data_lines = data[j].strip(' \n\r').split("\n")
		#print len(data_lines)
		for k in range(0,len(data_lines)-1):
			vecdata=data_lines[k].strip(' \n\r').split(' ')
			#print vecdata
			vecdatanew = [float(z) for z in vecdata]
			vecdatanew = (vecdatanew-np.mean(vecdatanew))/np.std(vecdatanew)
			scaler = MinMaxScaler(feature_range=(-1,1))
			vecarr = scaler.fit_transform(np.array(vecdatanew).reshape(-1,1))
			sz = vecarr.size
			vecdatanew = list(scaler.fit_transform(np.array(vecarr).reshape(-1,1)).reshape(1,sz)[0])
			#vecdatanew = 2.0*((vecdatanew-np.max(vecdatanew)))/-np.ptp(vecdatanew)-1
			vecdatastr = ' '.join(str(e) for e in vecdatanew)+"\n"
               		fw.write(vecdatastr)

	        fw.write(data_lines[len(data_lines)-1]+"\n")
        	fw.write("\n")
        fw.close()
	fr.close()

