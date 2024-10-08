import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# making folders
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data', outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# to keep count of each category
train_count = {name: 0 for name in inner_names}
test_count = {name: 0 for name in inner_names}

df = pd.read_csv('./fer2013.csv')

print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    # split pixel values and convert to 48x48 image
    pixels = np.array(df['pixels'][i].split(), dtype='uint8').reshape(48, 48)
    img = Image.fromarray(pixels)

    emotion = df['emotion'][i]
    emotion_label = inner_names[emotion]  # map to emotion name

    # train images
    if i < 28709:
        img.save(f'fer2013/train/{emotion_label}/im{train_count[emotion_label]}.png')
        train_count[emotion_label] += 1
    # test images
    else:
        img.save(f'fer2013/test/{emotion_label}/im{test_count[emotion_label]}.png')
        test_count[emotion_label] += 1

print("Done!")
