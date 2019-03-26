import sys,os
import glob
import re
import numpy as np
import random

mapping_file_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/data_videos_and_queries/"
data_file_name = mapping_file_extension+"old_sentences_50_part5.txt"
new_data_file_name = mapping_file_extension+"sentences_50_part5.txt"

fold = open(data_file_name,"r")
fnew = open(new_data_file_name,"a")

old_data = fold.read().lower().strip(' ').split('\n')
data=filter(lambda x: len(x.strip('').strip('\n\r'))>0, old_data)

i = 0
while(i<len(data)):
	newannotation = ""
	annotation = data[i+1].strip(' ')
	if(annotation[2]=='('):
		newannotation += "l1"
	else:
		newannotation += "r1"
	annotation_list = annotation.split(') (')
	newannotation = newannotation+" "+ annotation_list[1].replace(',','')+" "+annotation_list[3].replace(',','')
	fnew.write(data[i]+"\n"+newannotation+"\n\n")
	i = i+2
fold.close()
fnew.close()
