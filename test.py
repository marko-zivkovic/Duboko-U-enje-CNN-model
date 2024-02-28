from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D,Rescaling
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import mnist
from keras.models import Sequential,Model

import matplotlib.pyplot as plt
from keras import backend as K

import tensorflow as tf
import numpy as np
import os

np.random.seed(42) 
tf.random.set_seed(42)

#splitujemo dadaset
base_path = "Datasets\IAM_Words"
words_lista = []
words = open(f"{base_path}/words.txt","r").readlines()
for line in words:
    #print(f"{line}")
    if line [0] == "#": 
        continue
    if line.split(" ")[1] != "err": 
        words_lista.append(line)

len (words_lista)
np.random.shuffle(words_lista)

#splitujemo train/test
split_id = int(0.9 * len(words_lista))
train_primeri = words_lista[:split_id]
test_primeri = words_lista[split_id:]

val_split = int(0.5 * len(test_primeri))
validation_primeri = test_primeri[:val_split]
test_primeri = test_primeri[val_split:]

assert len(words_lista) == len(train_primeri) + len(validation_primeri) + len(test_primeri)
print(f"ukupan broj train primera {len(train_primeri)}" )
print(f"ukupan broj test primera {len(test_primeri)}" )
print(f"ukupan broj validation primera {len(validation_primeri)}" )

#pripremamo data inpust pipeline preko definisanih putanja slika
image_path = os.path.join(base_path, "wordsss")#wordsss
def get_image_paths_labels(samples):
    putanje = []
    correcred = []
    for(i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")
        image_name = line_split[0]
        p1 = image_name.split("-")[0]
        p2 = image_name.split("-")[1]
        img_path = os.path.join(image_path,p1,p1 + "-" + p2,image_name + ".png")
     
        if os.path.getsize(img_path):
            putanje.append(img_path) 
            correcred.append(file_line.split("\n")[0])
    

    return putanje,correcred

train_img, train_lab = get_image_paths_labels(train_primeri)
test_img, test_lab = get_image_paths_labels(test_primeri)
validation_img, validation_lab = get_image_paths_labels(validation_primeri)

train_lab_free = []
characters = set()
max_len = 0
for label in train_lab:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)
    max_len = max(max_len,len(label))
    train_lab_free.append(label)

print("max duzina ",max_len)
print("vokabular size ", len(characters))

def clean_lab(labels):
    free_labs = []
    for lab in labels:
        lab = lab.split(" ")[-1].strip()
        free_labs.append(lab)
    return free_labs

validation_lab_free = clean_lab(validation_lab)
test_lab_free = clean_lab(test_lab)

autotune = tf.data.AUTOTUNE
# char to integer
char_to_num = StringLookup(vokabular = list(characters), mask = None)
# integer to char
num_to_shar = StringLookup(vokabular = char_to_num.get_vokabulary(), mask = None, invert = True)

#resize
def distorzja_free_resize(image, ssize):
    w,h = ssize
    image = tf.image.resize(image, size = (h,w), p_a_r = True)
    #koliko padding-a je potrebno
    pad_h = h - tf.shape(image)[0]
    pad_w = w - tf.shape(image)[1]
    if pad_h % 2 != 0:
        visina = pad_h // 2
        pad_h_top = visina + 1
        pad_h_bottom = visina
    else:
        pad_h_top = pad_h_bottom = pad_h //2
    if pad_w % 2 != 0:
        sirina = pad_w // 2
        pad_w_l = sirina + 1
        pad_w_r = sirina
    else:
        pad_w_l = pad_w_r = pad_w //2
    image = tf.pad(image, paddings=[[pad_h_top,pad_h_bottom],[pad_w_l,pad_w_r],[0,0]],)

    image = tf.transpose(image, perm = [1,0,2])
    image = tf.image.flip_left_right(image)
    return image

#kombinacija svih podesavanja
batch_size = 64
padding_token = 99
image_w = 128
image_h = 32

def preproces(slika_path, img_size = (image_w,image_h)):
    image = tf.io.read_file(slika_path)
    image = tf.image.decode_png(image,1)
    image = distorzja_free_resize(image,img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image
def vectorize_lab(lab):
    lab = char_to_num(tf.strings.unicode_split(lab,input = "UTF-8"))
    duzina = tf.shape(lab)[0]
    pad_kolicina = max_len - duzina
    lab = tf.pad(lab,paddings=[[0,pad_kolicina]], const=padding_token)
    return lab
def process_images_labs(image_p,lab):
    image = preproces(image_p)
    lab = vectorize_lab(lab)
    return {"image":image, "label":lab}

def pripremi_dataset(img_path,labels):
    dataset = tf.data.Dataset.from_tensor_slices((img_path,labels)).map(
        process_images_labs, num_parallelcall = autotune
    )
    return dataset.batch(batch_size).cache().prefetch(autotune)
    
train_ds = pripremi_dataset(train_img,train_lab_free)
validation_ds = pripremi_dataset(validation_img,validation_lab_free)
test_ds = pripremi_dataset(test_img,test_lab_free)
x_train=train_img
y_train=train_lab_free
x_test=test_img
y_test=test_lab_free


#model
def CNN_model(image_height,image_width,num_classes,augment=True):
  model = Sequential()
  model.add(Rescaling(1.0/255,input_shape=(image_height, image_width, 1)))
  model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
  model.add(Dropout(rate=0.66))
  model.add(Flatten())
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

  return model

model = CNN_model(8,15,char_to_num)
#model.summary()
epochas = 100
model.fit(x_train,y_train,
          validation_split=0.2,epochs=epochas,validation_freq=2,workers=4)
obj = model.evaluate(x_test,y_test)
print(obj)