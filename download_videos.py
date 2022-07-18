import glob
import os
import random
from csv import reader
import glob

files = glob.glob('data/lvu_1.0/*/*.csv')
print('Total files : ', len(files))

all_ids = set()
for file in files:
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        if header != None:
            for row in csv_reader:
                if file.__contains__('view'):
                    youtube_id = row[0].split(' ')[1]
                else:
                    youtube_id = row[0].split(' ')[2]
                dest = f'/playpen-storage/mmiemon/lvu/data/mc_videos/{youtube_id}.mp4'
                #if not os.path.exists(dest):
                all_ids.add(youtube_id)

print('Total videos : ',len(all_ids))

for youtube_id in all_ids:
    dest = f'/playpen-storage/mmiemon/lvu/data/mc_videos/{youtube_id}.mp4'
    if not os.path.exists(dest):
        cmd = f'youtube-dl -f 18 --cookies youtube.com_cookies.txt -o {dest} https://www.youtube.com/watch?v={youtube_id}'
        os.system(cmd)




