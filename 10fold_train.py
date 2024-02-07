import os
for i in range(10):
    print('Fold', i)
    cmd = 'python CLIP_KMU.py --mode 1 --fold %d' %(i+1)
    os.system(cmd)
print("Train CLIP_KMU ok!")
os.system('pause')
