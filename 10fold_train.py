import os
for i in range(10):
    print('Fold', i)
    cmd = 'python CLIP_KMU.py --model Ourmodel --bs 64 --lr 0.003 --mode 1 --fold %d' %(i+1)
    os.system(cmd)
print("Train CLIP_KMU ok!")
os.system('pause')
