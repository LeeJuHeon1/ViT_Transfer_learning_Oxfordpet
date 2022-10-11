import tensorflow as tf
import tensorboard
import keras_cv
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers

from vit_keras import vit

print(tf.keras.__version__)

learning_rate = 0.01
training_epochs = 20
epoch = 10
batch_size = 16
num_classes = 37
input_shape = (224, 224)
vit_image_shape = (224,224)
buffer_size = 1024

"Load OxfordPet Dataset"
from dataloader import load_OxfordPet
train_ds, val_ds = load_OxfordPet(batch_size, buffer_size, input_shape)

def freeze_model(model, limit=  None):
    # handle negative indices
    if limit != None and limit < -1:
        limit = limit + len(model.layers) 
    # loop for all valid indices and mark the corresponding layer
    for index, layer in enumerate(model.layers):
        if limit != None and index > limit:
            break
        layer.trainable = False

def unfreeze_model(model):
    for index, layer in enumerate(model.layers):
        layer.trainable = True

model = vit.vit_b16(image_size=vit_image_shape, activation='sigmoid', pretrained=True, include_top=False, pretrained_top=False)   # pretrained vit model accuracy : 85.49%
print(model.summary())

freeze_model(model)

"Model top"
x = model.output                         # shape should be (bs=None, 7, 7, 2048)
x = Dropout(0.3)(x)     # shape should be (bs=None, 2048)
x = Dense(1024, activation='gelu')(x) # shape should be (bs=None, 1024)
x = BatchNormalization()(x)
y = Dense(num_classes, activation='softmax')(x) # shape should be (bs=None, 37)
model2 = Model(inputs=model.input, outputs=y)

model2.compile(
optimizer = Adam(learning_rate=learning_rate, epsilon=0.01, decay=0.0001),
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy']
)

# Set tensorboard
callback_tensorboard = keras.callbacks.TensorBoard(
    log_dir = './log',
    histogram_freq = 1,
    embeddings_freq = 1,
    )

trained_model = model2.fit(
    train_ds,
    batch_size=batch_size,
    epochs=training_epochs,
    validation_data=val_ds,
    callbacks=[callback_tensorboard],
    )

unfreeze_model(model)

model2.compile(
    optimizer = Adam(learning_rate=learning_rate/100, epsilon=0.01, decay=0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'] 
)

# Set tensorboard
callback_tensorboard2 = keras.callbacks.TensorBoard(
    log_dir = './logs',
    histogram_freq = 1,
    embeddings_freq = 1,
)

trained_model = model2.fit(
    train_ds,
    batch_size=batch_size,
    epochs=epoch,
    validation_data=val_ds,
    callbacks=[callback_tensorboard2],
)