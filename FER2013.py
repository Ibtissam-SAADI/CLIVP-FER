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
