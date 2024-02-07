import os
for i in range(10):
    print('Fold', i)
    cmd = 'python KDEFCLIPExtractImg+text.py --mode 1 --fold %d' %(i+1)
    os.system(cmd)
print("Train  ok!")
os.system('pause')
