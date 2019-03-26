import sys,os
import glob
import re
import numpy as np
import cv2
from vgg import *

frames_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/"
rst_queries_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/updated_queries"
frames_dirs = 50
frames_required = 60

dirextension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/frame_rst_mappings/"
if not os.path.exists(dirextension):
                os.makedirs(dirextension)

fp = open(rst_queries_extension+"/new.txt","r")
rst_queries = fp.read().split('\n')
fp.close()

start = int(sys.argv[1])
i =start
while(i < (start+2) and i<= frames_dirs):
	print(i)
	if os.path.exists(frames_extension+str(i)):
		fw = open(dirextension+str(i)+".txt","a")
		subdirfiles = glob.glob(frames_extension+str(i)+"/*.jpg")
		subdirfiles = sorted(list(subdirfiles))
		new_list = []
		if(len(subdirfiles) > frames_required):
			new_list = list(np.linspace(0,len(subdirfiles)-1,frames_required,dtype=int))	
		else:
			new_list = range(0,len(subdirfiles))
		for j in range(0,len(new_list)):
			video_file = subdirfiles[new_list[j]]
			# read an image
			#print video_file
			try:
    				im = cv2.resize(cv2.imread(video_file), (224, 224)).astype(np.float32)
			except:
				continue
			
			#exit(1)
    			im[:,:,0] -= 103.939
    			im[:,:,1] -= 116.779
    			im[:,:,2] -= 123.68
    			im = np.expand_dims(im, axis=0)

    			# build VGG_net without last fc layer and load weights
    			model = VGG_16()

    			# generate image feature
    			out_feature = model.predict(im)
			fw.write(str(list(out_feature[0]))+"\n")
			
		fw.write(rst_queries[i-1])
		fw.close()
	i = i+1
