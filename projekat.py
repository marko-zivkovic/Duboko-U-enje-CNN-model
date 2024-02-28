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
from PIL import Image
#np.random.seed(42) 
#tf.random.set_seed(42)


#splitujemo dadaset
base_path = "Datasets\IAM_Words"
words_lista = [] #lista putanju iz words.txt
words = open(f"{base_path}/words.txt","r").readlines()
for line in words:
    #print(f"{line}")
    if line [0] == "#": 
        continue
    if line.split(" ")[1] != "err": 
        words_lista.append(line)

ukupan_broj_reci = len (words_lista);
broj_reci = []
for recenica in words_lista:
    recenica = recenica.split(" ")
    rec = recenica[8].strip()
    broj_reci.append(rec)
broj_reci_jedinstven = list(set(broj_reci))#12218
print(f"ukupan_broj_reci: {len(broj_reci)} {len(broj_reci_jedinstven)}")

izabrane_reci = broj_reci_jedinstven[:100]
print(f"{izabrane_reci[:10]}")
nova_lista_reci = []#sadrzi putanju ime i word
for recenicaa in words_lista:
    recenicaa = recenicaa.split(" ")
    recc = recenicaa[8].strip()
    if recc in izabrane_reci:
        nova_lista_reci.append(recenicaa)
print(f"broj slika: nova_lista_reci: {len(nova_lista_reci)}")    

#splitujemo train/test
split_id = int(0.9 * len(nova_lista_reci))
train_primeri = words_lista[:split_id] #train x
test_primeri = words_lista[split_id:]  #test x

train_xx = []#slike bez resize
test_xx = []#slike bez resize
train_x = []
test_x = []
train_y = []
test_y = []
broj_klasa = len(izabrane_reci)
image_path = os.path.join(base_path, "wordsss")
def get_tarin_test(t_pr):
    putanje = []
    correcred = []
    for (i, file_line) in enumerate(t_pr):
        line_split = file_line
        line_split = line_split.split(" ")
        image_name = line_split[0]
        p1 = image_name.split("-")[0]
        p2 = image_name.split("-")[1]
        img_path = os.path.join(image_path,p1,p1 + "-" + p2,image_name + ".png")
     
        if os.path.getsize(img_path):
            putanje.append(img_path) 
            correcred.append(file_line[8].strip())

    return putanje,correcred # x,y

train_xx ,train_y = get_tarin_test(train_primeri)
test_xx ,test_y = get_tarin_test(test_primeri)
'''
#resize
def distorzja_free_resize(image, w,h):
    image = tf.image.resize(image, size = (h,w))
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
image_w = 70
image_h = 35

def preproces(slika_path, image_w, image_h):
    image = tf.io.read_file(slika_path)
    image = tf.image.decode_png(image,1)
    image = distorzja_free_resize(image,image_w,image_h)
    image = tf.cast(image, tf.float32) / 255.0
    return image

for line in train_xx:
    image = preproces(line, image_w,image_h)
    train_x.append(image)
for linee in test_xx:
    imagee = preproces(linee, image_w,image_h)
    test_x.append(imagee)

image1 = train_x[1]
image2 = train_x[2]
slika1 = Image.open(image1)
slika2 = Image.open(image2)
w , h = slika1.size
w1 , h1 = slika2.size
print(f"{w} {h} {w1} {h1}")
'''
def resize_slikeeee(putanja_do_slike, ciljana_sirina, ciljana_visina, putanja_sacuvane_slike):
    try:
        # Otvori sliku
        originalna_slika = Image.open(putanja_do_slike)
        # Vrši resize slike na osnovu ciljane širine i visine
        nova_slika = originalna_slika.resize((ciljana_sirina, ciljana_visina))
        if putanja_sacuvane_slike:
            # Sačuvaj rezultujuću sliku na novoj putanji
            nova_slika.save(putanja_sacuvane_slike)
        else:
            print(f"nista")

    except Exception as e:
        print(f"Greska prilikom obrade slike: {str(e)}")

# Primer upotrebe:
resize_slikeeee(train_xx[1],70,30,base_path)