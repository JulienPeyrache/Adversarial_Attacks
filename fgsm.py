import tensorflow as tf


def generate_adversary_image(model, image, label, eps = 2/255,targeted=False):
    #Cast image to create tensor
    image = tf.cast(image,tf.float32)

    with tf.GradientTape() as tape:
        # explicitly indicate that our image should be tacked for gradient updates
        tape.watch(image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = model(image)

        #Use MSE loss function
        loss = tf.keras.losses.categorical_crossentropy(label, pred) 


        # calculate the gradients of loss with respect to the image, then
        # compute the sign of the gradient

        gradient = tape.gradient(loss, image)
        signedGrad = tf.sign(gradient)
        # construct the image adversary
        if targeted:
            adversary = (image - (signedGrad * eps)).numpy() #On augmente la prob de ce label
        else:
            adversary = (image + (signedGrad * eps)).numpy() #On diminue la prob de ce label
        # return the image adversary to the calling function
        return adversary