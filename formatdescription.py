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
        classes.extend(["A facial expression with lowered and furrowed eyebrows, narrowed and glaring eyes, flared nostrils, a mouth either firmly pressed or snarling, and a tensed jaw."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([0]*len(f_files))
    elif classe == "disgust":
        classes.extend(["A facial expression with a wrinkled nose, raised upper lip, narrowed or squinting eyes, and a slightly open or curled lip."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([1]*len(f_files))
    elif classe == "fear":
        classes.extend(["A facial expression with wide-open eyes, raised and drawn together eyebrows, a tensed or slightly open mouth, and a generally stretched or elongated face."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([2]*len(f_files))
    elif classe == "happy":
        classes.extend(["A facial expression with wide, bright eyes, raised cheeks, a broad smile showing teeth, and relaxed eyebrows."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([3]*len(f_files))
    elif classe == "sadness":
        classes.extend(["A facial expression with downward pointing corners of the mouth, drooping eyelids, slightly furrowed brows, and a generally downwards and subdued look."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([4]*len(f_files))
    elif classe == "surprise":
        classes.extend(["A facial expression with raised eyebrows, wide-open eyes, a dropped jaw with the mouth open."]*len(f_files))
        all_files.extend(f_files)
        labels.extend([5]*len(f_files))

    dataframe = pd.DataFrame(
        {
            'images': all_files,
            'text': classes,
            'labels': labels,
        }
    )
    dataframe.to_csv('./des5data5.csv', index=False)
print(f"Length of the exported data: {len(dataframe)}")
