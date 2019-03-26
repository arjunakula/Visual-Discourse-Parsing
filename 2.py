import sys,os
import glob
import re

extension = "/home/arjunakula/Documents/frames_part1/*.txt"
dirextension = "/home/arjunakula/Documents/frames_part1/updated_queries/"
dirfiles = glob.glob(extension)
if not os.path.exists(dirextension):
                os.makedirs(dirextension)
RST_tags = ['sequence','background','NVC','NVR','enablement','elaboration','justify','purpose','contast','VC','VR','motivation','circumstance']
for i in dirfiles:
        tok = i.split("/")[-1].split(".txt")[0]
	new_name = dirextension+tok+'_new.txt'
        
        f1 = open(i, "r")
        f2 = open(new_name, "a")
        raw_content = f1.read().replace('l1','<LEFT_CONFIG>').replace('r1','<RIGHT_CONFIG>').replace('l2','<LEFT_DIR>').replace('r2','<RIGHT_DIR>').replace('b2','<BI_DIR>')
        for j in RST_tags:
		raw_content = raw_content.replace(j,'<'+j+'>')
	content = raw_content.split('\n')

        newcontent=filter(lambda x: len(x.strip('').strip('\n'))>0, content) 
	#print(newcontent)
        i = 0
        while(i < len(newcontent)):
                textlines = re.sub('\d.*\\)','',newcontent[i]).strip(' ').split('.')
		textlines=filter(lambda x: len(x.strip('').strip('\n'))>0, textlines)
		#print(textlines)
                annotations = newcontent[i+1].split(' ')
		annotations=filter(lambda x: len(x.strip('').strip('\n'))>0, annotations)
		new_annotations = ""
                if(len(textlines) == 3):
                	new_annotations = annotations[0]+' '+textlines[0]+' '+annotations[1]+' '+annotations[2]+' '+textlines[1]+' '+annotations[3]+' '+annotations[4]+' '+textlines[2]
                elif(len(textlines) == 2):
                	new_annotations = annotations[0]+' '+textlines[0]+' '+annotations[1]+' '+annotations[2]+' '+textlines[1]
		f2.write(new_annotations+"\n")
		#print(new_annotations)
		#exit(1)
		i = i+2
	f2.close()
	f1.close()

