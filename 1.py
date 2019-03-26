import sys,os
import glob

extension = "/home/arjunakula/Documents/frames_part1/*.mp4"
frames_extension = "/home/arjunakula/Documents/frames_part1/"
dirfiles = glob.glob(extension)

for i in dirfiles:
        tok = i.split("/")[-1].split(".mp4")[0]
        if not os.path.exists(frames_extension+tok):
    		os.makedirs(frames_extension+tok)
        vlc_command = '/usr/bin/vlc "'+ i+'" --video-filter=scene --vout=dummy --scene-ratio=3 --scene-path="'+frames_extension+tok+'" vlc://quit'
	#print vlc_command
	os.system(vlc_command)
