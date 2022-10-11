import tensorflow as tf
import os
import numpy as np
from oxfordpet_API import Transfer_Model
import matplotlib.pyplot as plt
from dataloader import load_OxfordPet # ,load_oxford

test_mode = 0     # custpm_image == 0
                  # evaluate_data == 1
if test_mode == 0:
# image load from file or numpy
    image = './dog2.jpg'
    test_path = './tuning_test_image/'

    test_img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    test_img_arr = tf.keras.preprocessing.image.img_to_array(test_img)
else:
    #Load OxfordPet Dataset
    train_ds, val_ds = load_OxfordPet(batch_size=16, shuffle_buffer_size=1024, input_shape=(384,384))


# load model
model_weights_path = './model_weights.h5'
model_weights_paths = './tmp/checkpoint'
vit_image_shape = (224,224)
num_classes = 37

vit_model = Transfer_Model(vit_image_shape=vit_image_shape, 
                           num_classes=num_classes, 
                           model_weights_path=model_weights_path
                           )

if test_mode == 0:
    print(vit_model.test_result(test_img_arr))
#     for folder, _, files in os.walk(test_path):
#         for file in files:
#             image_path = test_path + file
#             test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#             test_img_arr = tf.keras.preprocessing.image.img_to_array(test_img, dtype="float")
#             print(test_img_arr)
#             test_img_arr = vit_model.preprocess_image(test_img_arr)
#             test_img_pred = vit_model.model.predict(test_img_arr)
#             index = np.argsort(test_img_pred.flatten())[::-1][:5]
#             x = vit_model.labels_description[index]
#             out_prob = test_img_pred.flatten()[index]
#             print("True label : ",file)
#             print('Top1 : '+x[0],f"test accuracy = {round(out_prob[0] * 100, 2)}%")
#             print('Top2 : '+x[1],f"test accuracy = {round(out_prob[1] * 100, 2)}%")
#             print('Top3 : '+x[2],f"test accuracy = {round(out_prob[2] * 100, 2)}%")
#             print('Top4 : '+x[3],f"test accuracy = {round(out_prob[3] * 100, 2)}%")
#             print('Top5 : '+x[4],f"test accuracy = {round(out_prob[4] * 100, 2)}%")

#             if x[0] + '.jpg' == file:
#                 print("Predict result = True")
#             else:
#                 print("Predict result = False")
else:
    _, accuracy = vit_model.model.evaluate(val_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

# test model
# print(vit_model.test_result(test_mode))

""" Test data Image """
# vit_model.test_data(test_path)

# send output