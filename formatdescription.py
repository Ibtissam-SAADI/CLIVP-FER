import pandas as pd
import os

ck_path = 'datasets/KMUMTCN'

anger_path = os.path.join(ck_path, 'anger')
disgust_path = os.path.join(ck_path, 'disgust')
fear_path = os.path.join(ck_path, 'fear')
happy_path = os.path.join(ck_path, 'happy')
sadness_path = os.path.join(ck_path, 'sadness')
surprise_path = os.path.join(ck_path, 'surprise')

all_files = []
classes= []
labels = []

for chemin in [anger_path,disgust_path, fear_path, happy_path, sadness_path, surprise_path]:
    files = os.listdir(chemin)
    f_files = [chemin+'/'+f for f in files]

    classe = (chemin.split("\\")[-1])

    if classe == "anger":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        #classes.extend(["lowered, furrowed eyebrows, narrowed or glaring eyes, tightly pressed lips, tensed jaw muscles, and a tightened forehead."]*len(f_files))
        #classes.extend(["Eyebrows down and together, narrowed eyes, tight lips."]*len(f_files))
        #classes.extend(["furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([0]*len(f_files))
    elif classe == "disgust":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        #classes.extend(["a scrunched nose, raised upper lip, and squinting eyes, displaying repulsion without the furrowed brow and glare seen in anger."]*len(f_files))
        #classes.extend(["Wrinkled nose, raised upper lip, squinted eyes."]*len(f_files))
        #classes.extend(["a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([1]*len(f_files))
    elif classe == "fear":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        #classes.extend(["wide-open eyes, raised and drawn-together eyebrows, an open mouth, and a stretched, tense forehead, reflecting a sense of alarm or threat."]*len(f_files))
        #classes.extend(["Wide eyes, raised and drawn together eyebrows, slightly open mouth."]*len(f_files))
        #classes.extend(["raised eyebrows, parted lips, a furrowed brow, and a retracted chin."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([2]*len(f_files))
    elif classe == "happy":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        #classes.extend(["a smiling mouth with upturned corners, raised cheeks, and eyes that may crinkle at the corners, often indicating genuine joy or amusement."]*len(f_files))
        #classes.extend(["Upturned mouth corners, raised cheeks, crinkled eyes."]*len(f_files))
        #classes.extend(["a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([3]*len(f_files))
    elif classe == "sadness":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        #classes.extend(["downturned corners of the mouth, slightly furrowed eyebrows, drooping upper eyelids, and a general downward or softening of facial features, conveying a sense of melancholy or sorrow."]*len(f_files))
        #classes.extend(["Downturned mouth, drooping eyelids, furrowed brows."]*len(f_files))
        #classes.extend(["tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([4]*len(f_files))
    elif classe == "surprise":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        #classes.extend(["raised and curved eyebrows, wide-open eyes, and often a dropped jaw with an open mouth, signifying a sudden shock or astonishment."]*len(f_files))
        #classes.extend(["Raised eyebrows, wide-open eyes, open mouth."]*len(f_files))
        #classes.extend(["widened eyes, an open mouth, raised eyebrows, and a frozen expression."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([5]*len(f_files))

    dataframe = pd.DataFrame(
        {
            'images': all_files,
            'text': classes,
            'labels': labels,
        }
    )
    #dataframe.to_csv('./Descdata.csv', index=False)
    dataframe.to_csv('./simpledes06.csv', index=False)
print(f"Length of the exported data: {len(dataframe)}")