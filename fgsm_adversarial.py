from simpleCNN import build_model
from fgsm import generate_adversary_image
import tensorflow as tf
import numpy as np
import cv2


configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# On rajoute une dimension aux images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

trainY = tf.keras.utils.to_categorical(trainY, 10)
testY = tf.keras.utils.to_categorical(testY, 10)

#On construit le model

model = build_model(width=28,height=28,depth=1,nb_labels=10)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=1e-3),
	metrics=["accuracy"])

model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=1,verbose=1)

(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

nb_adv_input = 1000
reussi = 0
reussi_target = 0
reussi_target_sans = 0

for i in np.random.choice(np.arange(0, len(testX)), size=(nb_adv_input,)):
    image = testX[i]
    print(image.shape)
    label = testY[i]
    label_target = [0 for i in range(len(label))]
    label_target[1] = 1
    adversary = generate_adversary_image(model, image.reshape(1,28,28,1),label,eps=0.1,targeted=False)
    adversary_target = generate_adversary_image(model, image.reshape(1,28,28,1),label_target,eps=0.1,targeted=True) #targeted corredpond à l'application pour avoir une classe voulue ou non
                                                                                                       #remplacer par label par label_test pour viser une classe précise dans le cas targeted=True

    label_pred = model.predict(image.reshape(1,28,28,1))
    pred_adversary = model.predict(adversary)
    pred_adversary_target = model.predict(adversary_target)

    # scale both the original image and adversary to the range
    # [0, 255] and convert them to an unsigned 8-bit integers

    adversary = adversary.reshape((28, 28)) * 255
    adversary = np.clip(adversary, 0, 255).astype("uint8")
    image = image.reshape((28, 28)) * 255
    image = image.astype("uint8")

    # convert the image and adversarial image from grayscale to three
    # channel (so we can draw on them)

    image = np.dstack([image] * 3)
    adversary = np.dstack([adversary] * 3)
    # resize the images so we can better visualize them

    image = cv2.resize(image, (96, 96))
    adversary = cv2.resize(adversary, (96, 96))

    labelPred = label_pred[0].argmax()
    adversary_labelPred = pred_adversary[0].argmax()
    adversary_labelPred_target = pred_adversary_target[0].argmax()
    
    color = (0,255,0)

    reussi +=1
    if adversary_labelPred_target == 1 and labelPred != adversary_labelPred_target:
        reussi_target +=1

    if adversary_labelPred_target == 1 and labelPred != adversary_labelPred_target and adversary_labelPred != adversary_labelPred_target:
        reussi_target_sans +=1

    if labelPred != adversary_labelPred:
        color = (0, 0, 255)
        reussi -=1

    #cv2.putText(image, str(labelPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    #cv2.putText(adversary, str(adversary_labelPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    # stack the two images on the same axis

    #output = np.hstack([image, adversary])  
    #cv2.imshow("FGSM Adversarial Images", output)
    #cv2.waitKey(0)

print(reussi/nb_adv_input)
print(reussi_target/nb_adv_input)
print(reussi_target_sans/nb_adv_input)