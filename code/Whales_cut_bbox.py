import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing import image
import warnings
import os
warnings.simplefilter("ignore", category=DeprecationWarning)


print(os.listdir("input"))
TEST = 'input/humpback-whale-identification/test/'
LABELS = 'input/humpback-whale-identification/train.csv'
LABELS_MOD = 'input/humpback-whale-identification/train_no_new.csv'
LABELS_back = 'input/humpback-whale-identification/train_back.csv'
TRAIN = 'input/humpback-whale-identification/train/'

TEST_CROPPED = 'input/humpback-whale-identification/cropped_test/'
TRAIN_CROPPED = 'input/humpback-whale-identification/cropped_train/'

SAMPLE_SUB = 'input/humpback-whale-identification/sample_submission.csv'
BBOX = 'input/humpback-whale-identification/bounding_boxes.csv'

train = pd.read_csv(LABELS)


def load_image(filepath):
    img = image.load_img(filepath)
    img = img.convert(mode="RGB")
    return img


def save_image(filename, im):
    with open(filename, 'w') as f:
        im.save(f)


def cut_bboxes(images, bboxes, source_folder, target_folder):
    count = 0

    for i in images:
        filepath = source_folder + i
        img = load_image(filepath)
        bbox = bboxes.loc[i]
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']

        if not (x0 >= x1 or y0 >= y1):
            img = img.crop((x0, y0, x1, y1))
        target_filepath = target_folder + i
        save_image(target_filepath, img)

        if count % 500 == 0:
            print("Cropping image: ", count + 1, ", ", i)

        count += 1


whales_bbox = pd.read_csv(BBOX).set_index('Image')


train = os.listdir(TRAIN)
cut_bboxes(train, whales_bbox, TRAIN, TRAIN_CROPPED)

whales_test = os.listdir(TEST)
cut_bboxes(whales_test, whales_bbox, TEST, TEST_CROPPED)

