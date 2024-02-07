import numpy as np
import tensorflow as tf
MODEL = tf.keras.models.load_model("./Trained_Model/Tomato")
CLASS_NAMES = ['Bacterial-spot', 'Early-blight', 'Healthy', 'Late-blight',
               'Leaf-mold', 'Mosaic-virus', 'Septoria-leaf-spot', 'Yellow-leaf-curl-virus']

def TomatoModel(image):
    img_batch = np.expand_dims(image, 0)
    image_resized = tf.image.resize(img_batch, (256, 256))
    predictions =  MODEL.predict(image_resized)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return {
        'class': predicted_class,
        'accuracy': confidence,
        'StatusCode':200 
    }

