import os
import random
import numpy as np
import csv
import os
from shutil import copyfile

### Train
path_train ='human-protein-atlas-image-classification/train/'
files = sorted(os.listdir(path_train))
print(len(files))
#n = int(np.floor(0.1 * len(files)))
#print(n)
#indexes = np.random.choice(len(files),n).astype(int)
name_files = []
for i in range(0,len(files),4*5):
    name_files.extend([files[i],files[i+1],files[i+2],files[i+3]])
print(len(name_files))
final = []
for i in name_files:
    copyfile(path_train + str(i),'data/train/'+str(i))
    final.append(i[:36])

with open("human-protein-atlas-image-classification/train.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = [row for row in reader if row['Id'] in final]

with open('test.csv', 'w') as f:
    for r in rows:
        for key in r.keys():
            f.write("%s,"%(r[key]))
        f.write('\n')
