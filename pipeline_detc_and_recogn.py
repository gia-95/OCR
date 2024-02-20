import keras_ocr
import tensorflow as tf
import matplotlib.pyplot as plt
import time

start_time = time.time()


# Upload image
path_image = 'documenti/sanitaria_retro.jpg'
image = keras_ocr.tools.read(path_image)


# Build Detector
detector = keras_ocr.detection.Detector(weights='clovaai_general')


# Build Recognizer
recognizer = keras_ocr.recognition.Recognizer()
recognizer.model = tf.keras.models.load_model('modelli_allenati/recogn_model_colab_05')
recognizer.compile()


# Fetch buonding-boxes image (detector) 
print("Detect parole...")
boxes = detector.detect(images=[image])[0]
drawn = keras_ocr.tools.drawBoxes(image=image, boxes=boxes)
plt.imshow(drawn)


# Recognize found bound-boxes
parole = []
for idx, box in enumerate(boxes) :
    box = box.astype('int64')
    img_crop = image[box[0][1]:box[2][1], box[0][0]:box[2][0]] # Ritaglia dall'immagine la parola corrente     
    predicted_word = recognizer.recognize(img_crop)
    parole.append(predicted_word)


print("\nParole trovate:", parole)
print("\nTempo impiegato: %s secondi" % (time.time() - start_time))
