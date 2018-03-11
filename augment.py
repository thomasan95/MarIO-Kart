import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
path = 'samples'
for race in os.listdir(path):
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
        #print row
        row = row.split(',')
        if len(row) >1:
            row[1]=str(-float(row[1]))
            row[2]=str(-float(row[2]))
            row[0]= newPath+'/'+csv_file
        row = ','.join(row)
        times.append(row)
    new_file = '\n'.join(times)
    newFile = open(newPath+'/'+csv_file,'w')
    newFile.write(new_file)
    newFile.close()
