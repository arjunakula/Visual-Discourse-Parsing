import sys,os
import glob

extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/data_videos_and_queries/videos_part_tmp/*.mp4"
frames_extension = "/home/arjunakula/temp_XAI/naacl2018_expt/NAACL2018_data/frames_part_tmp/"
dirfiles = glob.glob(extension)

for i in dirfiles:
        tok = i.split("/")[-1].split(".mp4")[0]
        if not os.path.exists(frames_extension+tok):
    		os.makedirs(frames_extension+tok)
        vlc_command = '/usr/bin/vlc "'+ i+'" --video-filter=scene --vout=dummy --scene-ratio=3 --scene-path="'+frames_extension+tok+'" vlc://quit'
	os.system(vlc_command)
