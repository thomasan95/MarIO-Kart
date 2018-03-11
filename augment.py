import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
path = 'samples'

def main():
    #mirror_images()
    races = filter( lambda f: not f.startswith('.'), os.listdir(path))
    for race in races:
        mirror_images_2(race)
        start_frames(race)

def mirror_images():
    races = filter( lambda f: not f.startswith('.'), os.listdir(path))
    for race in races:
        if race =='.keep':
            continue
        else:
            imagePath = path+'/'+race
            if not os.path.isdir('aug'):
                os.mkdir('aug')
            newPath = 'aug/'+race+'mirrored'
            if not os.path.isdir(newPath):
                os.mkdir(newPath)

        for image in os.listdir(imagePath):
            ext = os.path.splitext(image)[-1].lower()
            if not ext =='.png':
                csv_file = image
                continue
            img = np.fliplr(plt.imread(imagePath+'/'+image))
            plt.imsave(newPath+'/'+image, img)
        string = open(imagePath+'/'+csv_file)
        rows = string.read()
        rows = rows.split('\n')
        times = []
        for row in rows:
            row = row.split(',')
            if len(row) >1:
                row[1]=str(-float(row[1]))
                row[2]=str(float(row[2]))
                row[0]= newPath+'/'+csv_file
            row = ','.join(row)
            times.append(row)

        start_frames = rows[:30]*20
        times.extend(start_frames)

        new_file = '\n'.join(times)
        newFile = open(newPath+'/'+csv_file,'w')
        newFile.write(new_file)
        newFile.close()

def mirror_images_2(race):
    assert isinstance(race,str)

    race_path = 'samples/'+race
    race_mirror_path = 'samples_mirrored/'+race+'_mirrored'
    if not os.path.isdir('samples_mirrored'):
        os.mkdir('samples_mirrored')
    if not os.path.isdir(race_mirror_path):
        os.mkdir(race_mirror_path)

    # Get all images (.png) from race folder
    files = filter( lambda f: not f.startswith('.'), os.listdir(race_path))
    for file in files:
        #ignore hidden files
        ext = os.path.splitext(file)[-1].lower()
        if not ext=='.png':
            csv_file = file
            continue
        # flip image across y-axis
        new_image = np.fliplr(plt.imread(race_path+'/'+file))
        plt.imsave(race_mirror_path+'/'+file, new_image)

    # Edit the commands recorded from the controller
    
    with open(race_path+'/'+csv_file,'r') as f:
        rows = f.read().splitlines()

    new_commands = []
    for row in rows:
        row = row.split(',')
        if len(row) > 1:
            row[1]=str(-float(row[1]))
            row[0]= race_mirror_path+'/'+row[0].split('/')[-1]
        row = ','.join(row)
        new_commands.append(row)

    with open(race_mirror_path+'/'+csv_file, 'w') as f:
        values = '\n'.join(new_commands)
        f.write(values)


def start_frames(race):
    assert isinstance(race,str)

    race_path = 'samples/'+race
    start_frame_path = 'start_frames/'+race+'_start'

    # make directory for start frames
    if not os.path.isdir('start_frames'):
        os.mkdir('start_frames')

    # make directory for specific racfes
    if not os.path.isdir(start_frame_path):
        os.mkdir(start_frame_path)

    
    # ignore hidden files
    files = filter( lambda f: not f.startswith('.'), os.listdir(race_path))
    for file in files:#os.listdir(race_path):
        # Find CSV file
        ext = os.path.splitext(file)[-1].lower()
        if ext =='.csv':
            csv_file = file
            break

    with open(race_path+'/'+csv_file,'r') as f:
        rows = f.read().splitlines()

    new_commands=[]
    for row in rows[:30]:
        row = row.split(',')
        if len(row) >1:
            image_to_save = plt.imread(row[0])
            image_path = start_frame_path+'/'+row[0].split('/')[-1]
            plt.imsave(image_path ,image_to_save)
            row[0]= image_path
        row = ','.join(row)
        new_commands.append(row)

    new_commands = new_commands*30
    # write to new CSV file
    with open(start_frame_path+'/'+csv_file, 'w') as f:
        values = '\n'.join(new_commands)
        f.write(values)


if __name__=="__main__":
    main()



