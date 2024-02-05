# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr 12 10:33:35 2023

# @author: pret
# """

# # create data and label for FER2013
# # labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# import csv
# import os
# import numpy as np
# import h5py

# file = 'fer2013/fer2013.csv'

# # Creat the list to store the data and label information
# Training_x = []
# Training_y = []
# PublicTest_x = []
# PublicTest_y = []
# PrivateTest_x = []
# PrivateTest_y = []

# datapath = os.path.join('data','ferdata2013.h5')
# if not os.path.exists(os.path.dirname(datapath)):
#     os.makedirs(os.path.dirname(datapath))

# with open(file,'r') as csvin:
#     data=csv.reader(csvin)
#     for row in data:
#         if row[-1] == 'Training':
#             temp_list = []
#             for pixel in row[1].split( ):
#                 temp_list.append(int(pixel))
#             I = np.asarray(temp_list)
#             Training_y.append(int(row[0]))
#             Training_x.append(I.tolist())

#         if row[-1] == "PublicTest" :
#             temp_list = []
#             for pixel in row[1].split( ):
#                 temp_list.append(int(pixel))
#             I = np.asarray(temp_list)
#             PublicTest_y.append(int(row[0]))
#             PublicTest_x.append(I.tolist())

#         if row[-1] == 'PrivateTest':
#             temp_list = []
#             for pixel in row[1].split( ):
#                 temp_list.append(int(pixel))
#             I = np.asarray(temp_list)

#             PrivateTest_y.append(int(row[0]))
#             PrivateTest_x.append(I.tolist())

# print(np.shape(Training_x))
# print(np.shape(PublicTest_x))
# print(np.shape(PrivateTest_x))

# datafile = h5py.File(datapath, 'w')
# datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
# datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
# datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
# datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
# datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
# datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
# datafile.close()

# print("Save data finish!!!")

import csv
import os
import numpy as np

file = 'fer2013.csv'

# Text descriptions for each emotion class
text_descriptions = {
    0: 'A facial expression with lowered and furrowed eyebrows, narrowed and glaring eyes, flared nostrils, a mouth either firmly pressed or snarling, and a tensed jaw.',
    1: 'A facial expression with a wrinkled nose, raised upper lip, narrowed or squinting eyes, and a slightly open or curled lip.',
    2: 'A facial expression with wide-open eyes, raised and drawn together eyebrows, a tensed or slightly open mouth, and a generally stretched or elongated face.',
    3: 'A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows.', 
    4: 'A facial expression with downward pointing corners of the mouth, drooping eyelids, slightly furrowed brows, and a generally downwards and subdued look.',
    5: 'A facial expression with raised eyebrows, wide-open eyes, a dropped jaw with the mouth open.',
    6: 'A facial expression with relaxed eyebrows, eyes in a natural and unfocused state, a closed or slightly open mouth without tension.'
}

# Create directories for CSV files
if not os.path.exists('data'):
    os.makedirs('data')

# Process and save data to CSV files
with open(file, 'r') as csvin, \
     open('data/Training1.csv', 'w', newline='') as train_csv, \
     open('data/PublicTest1.csv', 'w', newline='') as public_csv, \
     open('data/PrivateTest1.csv', 'w', newline='') as private_csv:

    data = csv.reader(csvin)
    train_writer = csv.writer(train_csv)
    public_writer = csv.writer(public_csv)
    private_writer = csv.writer(private_csv)

    # Write headers
    train_writer.writerow(['pixels', 'text', 'label'])
    public_writer.writerow(['pixels', 'text', 'label'])
    private_writer.writerow(['pixels', 'text', 'label'])

    next(data)  # Skip header row

    for row in data:
        pixels = ' '.join(row[1].split())
        label = int(row[0])
        text = text_descriptions[label]

        if row[-1] == 'Training':
            train_writer.writerow([pixels, text, label])

        elif row[-1] == "PublicTest":
            public_writer.writerow([pixels, text, label])

        elif row[-1] == 'PrivateTest':
            private_writer.writerow([pixels, text, label])

print("Data saved in CSV format!")
