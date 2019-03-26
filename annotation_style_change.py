import sys,os
import glob
import re

extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part6/*.txt"
dirextension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part6/updated_queries/"
dirfiles = glob.glob(extension)
if not os.path.exists(dirextension):
                os.makedirs(dirextension)
RST_tags = ['sequence','background','nvc','nvr','enablement','elaboration','justify','purpose','contrast','vc','vr','motivation','circumstance','cause','evaluation','interpretation']

for i in dirfiles:
        tok = i.split("/")[-1].split(".txt")[0]
	new_name = dirextension+tok+'_new.txt'
        
        f1 = open(i, "r")
        f2 = open(new_name, "a")
        raw_content = f1.read().lower().replace('l1','<LEFT_CONFIG>').replace('r1','<RIGHT_CONFIG>').replace('l2','<LEFT_DIR>').replace('r2','<RIGHT_DIR>').replace('b2','<BI_DIR>')
	content = raw_content.split('\n')

        newcontent=filter(lambda x: len(x.strip(' ').strip('\n\r'))>0, content) 
        i = 0
        while(i < len(newcontent)):
                textlines = re.sub('\d.*\\)','',newcontent[i]).strip(' \n\r').split('.')
		textlines=filter(lambda x: len(x.strip(' ').strip('\n\r'))>0, textlines)
		#print(textlines)
        	for j in RST_tags:
			newcontent[i+1] = newcontent[i+1].lower().replace(j.lower(),'<'+j.lower()+'>')
                annotations = newcontent[i+1].strip(' \n\r').split(' ')
		annotations=filter(lambda x: len(x.strip(' ').strip('\n\r'))>0, annotations)
		new_annotations = ""
                if(len(textlines) == 3):
                	new_annotations = annotations[0]+' '+textlines[0]+' '+annotations[1]+' '+annotations[2]+' '+textlines[1]+' '+annotations[3]+' '+annotations[4]+' '+textlines[2]
                elif(len(textlines) == 2):
                	new_annotations = annotations[0]+' '+textlines[0]+' '+annotations[1]+' '+annotations[2]+' '+textlines[1]
		f2.write(new_annotations+"\n")
		#exit(1)
		i = i+2
	f2.close()
	f1.close()

