import tensorflow as tf
import numpy as np
from vit_keras import vit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class Transfer_Model:
    def __init__(self, vit_image_shape, num_classes, model_weights_path):
        super(Transfer_Model, self).__init__()
        self.vit_image_shape = vit_image_shape
        self.num_classes = num_classes
        
        self.labels_description = np.array([
            'abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
            'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua',
            'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
            'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'maine_coon',
            'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll',
            'russian_blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese',
            'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier',
            ])
        
        model = vit.vit_b16(image_size=self.vit_image_shape, activation='sigmoid', pretrained=True, include_top=False, pretrained_top=False)
        
        x = model.output                         # shape should be (bs=None, 7, 7, 2048)
        x = Dropout(0.3)(x)     # shape should be (bs=None, 2048)
        x = Dense(512, activation='swish')(x) # shape should be (bs=None, 1024)
        x = BatchNormalization()(x)
        y = Dense(self.num_classes, activation='softmax')(x) # shape should be (bs=None, 37)
        self.model = Model(inputs=model.input, outputs=y)
        self.model.load_weights(model_weights_path)
    
    def preprocess_image(self, image):
        test_image = image/255.0
        test_img_arr = np.expand_dims(test_image, axis=0)
        return test_img_arr
        
    def test_result(self, x, top_k = 5):
        """_summary_

        Args:
            x (_type_): ndarray 3d shape cannel last
        """
        out = self.model.predict(self.preprocess_image(x))

        index = np.argsort(out.flatten())[::-1][:top_k]
        label_name = self.labels_description[index]
        out_prob = out.flatten()[index]
        
        # index = np.array([])
        # for i, letter in enumerate(out_prob):
        #     index = np.append(index, i).astype(int)
        return out_prob, label_name