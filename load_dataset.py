import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

from tensorflow.keras.preprocessing import image_dataset_from_directory




train_dir = "dataset_videos\\train"
IMG_SIZE = (224, 224)
# Batches changes how many images are saved in each data element, if its 20 there will be (number elements in image folder / 20) dataset elements
# Second element is which calss it belongs to elem[1], can be hard to see if you set shuffle=True

path2 = "dataset_videos\\train\\01April_2010_Thursday_heute-6694"
path, dirs, files = next(os.walk(train_dir))
print("Printing directories:")
print(dirs)

#file_count = len(files)

# Idea: use image_dataset_from_directory on each folder and zip datasets afterwards, this way each element might be able to have varying batch size, test zip on small example first!
BATCH_SIZE = 1

# Can resize with image_size=IMG_SIZE
train_dataset = image_dataset_from_directory(train_dir,
                                            shuffle=False,
                                            batch_size=BATCH_SIZE)
class_names = train_dataset.class_names

placeholder_lang = []

for i in range(30):
    placeholder_lang.append(i)

placeholder_lang = np.array(placeholder_lang)

merged_dataset = []
images = []
count = 0
for elem in train_dataset:
    image = elem[0].numpy()
    images.append(image)
    label = int(elem[1].numpy())
    
    sentence = placeholder_lang[label]
    combined_tuple = (image, sentence)
    merged_dataset.append(combined_tuple)

as_np = np.array(merged_dataset, dtype=object)
print(as_np.shape)
imgTensor = tf.convert_to_tensor(images)

#print(type(merged_dataset))


#dataset9 = tf.data.Dataset.from_tensor_slices(merged_dataset)


dataset = tf.data.Dataset.from_tensor_slices(images)
#dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6], [7, 8]])
dataset = dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
#lambdafunc = lambda value, labels:  
lambdafunc = lambda x: x + 1  
test = 3

def upper_case_fn(t: tf.Tensor):
  return t.numpy() + placeholder_lang[t.numpy()]

dataset = dataset.map(lambda x: tf.py_function(func=upper_case_fn,
          inp=[x], Tout=tf.int32))


#def add_tuple(t: tf.Tensor, t2: tf.Tensor):
#    test = (t, t2)
#    return test
#  return (((t, t2), placeholder_lang[t2.numpy()]))
#dataset2 = train_dataset.map(lambda x, y: tf.py_function(func=add_tuple,
#          inp=[x,y], Tout=tuple()))

#for i in dataset:
#    print(i.numpy())

#print(list(dataset.as_numpy_iterator()))


#dataset = train_dataset.map(lambda x: (x, (x, placeholder_lang[int(x[1].numpy())]))
#print(original_list)
#print(file_count)
#print(len(train_dataset))


# Get number of videos in folder
#irr_path, irr_dirs, files = next(os.walk(path2))
file_count2 = 2 #len(files)
# Convert folder to tensorflow dataset
#dataset_folder = image_dataset_from_directory(path2, shuffle=False, batch_size=1)
# Convert dataset to numpy array
folder_as_numpy = []
#for elem in dataset_folder:
#    print("test")
    #folder_as_numpy.append(elem[0].numpy())


#img_folder_as_numpy = load_folder_as_numpy(path2)

#print(len(img_folder_as_numpy))