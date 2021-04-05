import numpy as np
from PIL import Image
import glob
import os
import tensorflow as tf

filelist = glob.glob('dataset_videoes\\train\\01April_2010_Thursday_heute-6694\\*.png')
filelist2 = glob.glob('dataset_videoes\\train\\01April_2010_Thursday_heute-6695\\*.png')

train_dir = "dataset_videoes\\train"
path, dirs, files = next(os.walk(train_dir))

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

np_array_imgs_all_dirs = []



for dir_path in dirs:
    filelist = glob.glob('dataset_videoes\\train\\' + dir_path + '\\*.png')
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    np_array_imgs_all_dirs.append(x)


x2 = np.array([np.array(Image.open(fname)) for fname in filelist2])
test = np.array([x, x2])
tf.convert_to_tensor(x)
tf.convert_to_tensor(x2)

print(x.shape) # (53, 224, 224, 3)
# perfect?
print(test.shape) # (2,) 
print(type(test))
print(test[1].shape) # (90, 224, 224, 3)
print(np_array_imgs_all_dirs[2].shape)
print(len(np_array_imgs_all_dirs))

conv_test = np.array(test)
print(type(conv_test))
print(conv_test.shape) # (2,) 

print(conv_test[1].shape) # (90, 224, 224, 3)

lang_placeholder = []

for i in range(30):
    lang_placeholder.append(i)

tupled_lists = (np_array_imgs_all_dirs, lang_placeholder)

nested_list = [np_array_imgs_all_dirs, lang_placeholder]

print("Look here")
print(type(np_array_imgs_all_dirs[0][0]))


dataset = tf.data.Dataset.from_generator(lambda: (np_array_imgs_all_dirs, lang_placeholder) , tf.int32)

count = 0
for element in dataset:
    #print(element.shape)
    count += 1
    print(count)
print("final message")
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()


#data_tensor = tf.ragged.constant(conv_test[1]) # wtf it never finnishes this operation
