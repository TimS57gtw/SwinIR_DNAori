import copy
import os
import random
from SPM_Filetype import SPM
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
# the standard network layer
import tensorflow.keras.utils
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers  # to choose more advanced optimizers like 'adam'
from tqdm import tqdm
import numpy as np
from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import *

import pydot

from PIL import Image, ImageOps

#from tqdm import tqdm

import matplotlib.pyplot as plt  # for plotting
import matplotlib

matplotlib.rcParams['figure.dpi'] = 200  # highres display

# for subplots within subplots:
from matplotlib import gridspec

# for nice inset colorbars: (approach changed from lecture 1 'Visualization' notebook)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# for updating display
# (very simple animation)
from time import sleep

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# tf.config.experimental_run_functions_eagerly(True)

# ### Functions

BATCHSIZE=30



# backpropagation and training routines
# these are now just front-ends to keras!
def get_layer_activation_extractor(network):
    return(Model(inputs=network.inputs,
                            outputs=[layer.output for layer in network.layers]))

def get_layer_activations(network, y_in):
    """
    Call this on some test images y_in, to get the intermediate
    layer neuron values. These are returned in a list, with one
    entry for each layer (the entries are arrays).
    """
    extractor=get_layer_activation_extractor(network)
    layer_features = extractor(y_in)
    return(layer_features)

def print_layers(network, y_in):
    """
    Call this on some test images y_in, to get a print-out of
    the layer sizes. Shapes shown are (batchsize,pixels,pixels,channels).
    After a call to the visualization routine, y_target will contain
    the last set of training images, so you could feed those in here.
    """
    layer_features=get_layer_activations(network,y_in)
    for idx,feature in enumerate(layer_features):
        s=np.shape(feature)
        print("Layer "+str(idx)+": "+str(s[1]*s[2]*s[3])+" neurons / ", s)

# ## Exercise: Extend this code so as to be able to use more advanced keras activation functions for the layers!
def preprocess_img_csv(f, folder, targetname, rename=False):
    text = ""
    with open(f, "r") as file:
        for line in tqdm(file):
            if line.startswith("filename"):
                continue
            line = line.strip()
            parts = line.split(",")
            fp = parts[0]
            #for i in range(len(fp)):
            #    print(i, fp[i])
            num = int(fp[10:-4])
            #print(num)

            name = os.path.join(folder, "bild", "Image" + str(num).zfill(6) + ".png")
            name2 = os.path.join(folder, "bild", "bild", "Image" + str(num).zfill(6) + ".png")

            if rename:
                os.rename(os.path.join(folder, "bild", fp), name2)
            text += name + "," + str(parts[1]) + "\n"



    with open(targetname, "w") as f:
        f.write(text)

def test():
    def train_csv(Net, filename, epochs=30):
        data = []
        train_pct = 0.7
        val_pct = 0.2
        test_pct = 0.1
        steps = 10000
        epochs = epochs

        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                parts = line.split(",")
                data.append(([float(parts[0])], [float(parts[1])]))

        np.random.shuffle(data)
        size = len(data)
        train_set = []
        val_set = []
        test_set = []

        for i in range(size):
            if i/size < train_pct:
                train_set.append(data[i])
            elif i/size < val_pct:
                val_set.append(data[i])
            else:
                test_set.append(data[i])

        def dataloader_train():
            input = []
            target = []
            for i in range(BATCHSIZE):
                pair = random.choice(train_set)
                input.append(pair[0])
                target.append(pair[1])

            return np.array(input), np.array(target)

        def dataloader_val():
            input = []
            target = []
            for i in range(BATCHSIZE):
                pair = random.choice(val_set)[0]
                input.append(pair[0])
                target.append(pair[1])

            return np.array(input), np.array(target)

        def dataloader_test():
            input = []
            target = []
            for i in range(BATCHSIZE):
                pair = random.choice(test_set)[0]
                input.append(pair[0])
                target.append(pair[1])

            np.array(input), np.array(target)

        def batch_loader_test():
            random.shuffle(test_set)
            i = 0
            while True:
                inp = []
                tar = [] #Test
                for j in range(BATCHSIZE):
                    pair = test_set[i]
                    inp.append(pair[0])
                    tar.append(pair[1])
                    i += 1
                    if i == len(test_set):
                        yield np.array(inp), np.array(tar), True
                        return
                yield np.array(inp), np.array(tar), False


        costs = []
        for ep in tqdm(range(epochs), desc="Training"):
            loader = batch_loader_test()
            batch_cost = 0
            i = 0

            for inp, tar, finish in loader:
                i+= 1
                batch_cost += Net.train_on_batch(inp, tar)[0]

            costs.append(batch_cost/i)

        plt.plot(costs)
        plt.title("Costs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        xs = np.linspace(-2, 2, 500)
        ys = Net.predict(xs, batch_size=BATCHSIZE)
        f = lambda x: 0.5*np.sin(x) * x
        tar = [f(x) for x in xs]

        plt.plot(xs, ys, label="Predict")
        plt.plot(xs, tar, label="Target")
        plt.legend()
        plt.title("Result")
        plt.show()



    Model = Sequential()
    Model.add(Dense(64, activation="sigmoid"))
    Model.add(Dense(256, activation="sigmoid"))
    Model.add(Dense(256, activation="sigmoid"))
    Model.add(Dense(64, activation="sigmoid"))
    Model.add(Dense(1))

    steps = 10000

    optimizer = optimizers.Adam(learning_rate=0.01)

    Model.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    f = lambda x: 0.5*np.sin(x) * x

    with open("test.csv", "w") as file:
        for i in range(10000):
            x = np.random.normal(0, 1)
            file.write(str(x) + "," + str(f(x)) + "\n")

    train_csv(Model, "test.csv")
    print("Soll: ", [f(-1), f(0), f(1)])
    print("Ist: ", Model.predict([-1, 0, 1]))


def train_img(imgs=os.path.join("Dataset3", "bild"), label_csv="label.csv", epochs=30,lr=None, start_lr=0.001, end_lr=0.001):
    def train_csv(Net, img_path, filename, epochs=11):
        epochs = epochs

        labels = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.split(",")
                tar = float(parts[1].strip())
                tar = int(1000 * tar - 500)
                labels.append(tar)



        train_ds = tensorflow.keras.utils.image_dataset_from_directory(
            img_path,
            labels=labels,
            label_mode="int",
            class_names=None,
            color_mode="grayscale",
            batch_size=BATCHSIZE,
            image_size=(256, 256),
            shuffle=True,
            seed=69,
            validation_split=0.2,
            subset="training",
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False,
        )

        val_ds = tensorflow.keras.utils.image_dataset_from_directory(
            imgs,
            labels=labels,
            label_mode="int",
            class_names=None,
            color_mode="grayscale",
            batch_size=BATCHSIZE,
            image_size=(256, 256),
            shuffle=True,
            seed=69,
            validation_split=0.2,
            subset="validation",
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False,
        )
        #for elem in train_ds:
        #    print_layers(Net, elem)
        #    break

        train_losses = []
        val_losses = []

        if start_lr != end_lr:
            adjust_lr = True
            beta = (end_lr/start_lr)**(1/epochs)
        else:
            adjust_lr=False

        lr = start_lr
        for ep in range(epochs):
            print("Epoch: {}/{}".format( ep+1, epochs))
            if adjust_lr:
                K.set_value(Net.optimizer.learning_rate, lr)
            hist = Net.fit(train_ds, epochs=1)
            train_losses.append(hist.history['loss'])
            loss, accuracy = Net.evaluate(val_ds)
            val_losses.append(loss)
            if adjust_lr:
                lr *= beta
            if (ep+1) % 10 == 0:
                Net.save(os.path.join(os.getcwd(), "Width", "Models", "model{}.pth".format(str(ep+1).zfill(4))))


        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.title("Losses")
        plt.savefig("Losses.png")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(2*np.pi, fill_mode="constant"),
    ])

    Model = Sequential()
    Model.add(data_augmentation)
    Model.add(Conv2D(8, 5,activation='relu', padding="same"))
    Model.add(BatchNormalization())
    Model.add(MaxPooling2D(pool_size=(2,2)))
    Model.add(Conv2D(16, 3, activation='relu', padding="same"))
    Model.add(BatchNormalization())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(8, 3, activation='relu', padding="same"))
    Model.add(BatchNormalization())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Conv2D(8, 3, activation='relu', padding="same"))
    Model.add(BatchNormalization())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Flatten())
    Model.add(Dense(1024, activation="sigmoid"))
    Model.add(BatchNormalization())
    Model.add(Dense(768, activation="sigmoid"))
    Model.add(BatchNormalization())
    Model.add(Dense(128))
    Model.add(Dense(1))

    Model.build(input_shape=(BATCHSIZE, 256, 256, 1))

    optimizer = optimizers.RMSprop(learning_rate=0.00005)

    Model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    Model.summary()

    tf.keras.utils.plot_model(Model, to_file='model.png')

    train_csv(Model, imgs, label_csv, epochs=epochs)



    lines = []
    with open(label_csv, "r") as f:
        for line in f:
            lines.append(line)

    fig, ax = plt.subplots(1, 3)


    for i in range(3):
        line = random.choice(lines)
        parts = line.strip().split(",")
        fn = parts[0].split("\\")
        fnb = os.path.join(fn[0], "bild", fn[1], fn[2])

        lbl = float(parts[1])

        im, im2 = prepare_img(fnb)
        pred = Model(im)[0][0]
        pred = (pred +500) / 1000
        #pred = 5
        ax[i].imshow(im2, cmap="gray")
        ax[i].set_title("L: {:.3f}, P: {:.3f}".format(lbl, pred))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.savefig("Examples.png")

def eval_medium_err(test_csv, img_path, Model=None, model_path=None, batchsize=32):
    batchsize = 10

    labels = []
    with open(test_csv, "r") as f:
        for line in f:
            parts = line.split(",")
            tar = float(parts[1].strip())
            tar = int(1000 * tar - 500)
            labels.append(tar)


    val_ds = tensorflow.keras.utils.image_dataset_from_directory(
        img_path,
        labels=labels,
        label_mode="int",
        class_names=None,
        color_mode="grayscale",
        batch_size=batchsize,
        image_size=(256, 256),
        shuffle=True,
        seed=69,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    # for elem in train_ds:
    #    print_layers(Net, elem)
    #    break

    train_losses = []
    val_losses = []

    if Model is None:
        Model = tf.keras.models.load_model(model_path)




    errors = []
    for inp, tar in val_ds:
        out = Model(inp)
        out_pct = (out+500)/10
        tar_pct = (tar+500)/10
        for i in range(len(out_pct)):
            print("Abw: ", (float(out_pct[i]) - float(tar_pct[i])))
            print("Rel: ", 100 * (float(out_pct[i]) - float(tar_pct[i]))/float(out_pct[i]))
            errors.append(100 * (float(out_pct[i]) - float(tar_pct[i]))/float(out_pct[i]))

        #print("Out: ", out_pct)
        #print("Tar: ", tar_pct)

    cases = len(errors)
    summe = 0
    for x in errors:
        summe += np.abs(x)
    avg =summe/ cases


    plt.hist(errors, bins=3*int(np.sqrt(len(errors))))
    plt.title("Avg Error: {:.3f} %".format(avg))
    plt.xlabel("relative distance to target (in percent)")
    plt.ylabel("# of test cases")
    plt.savefig("AVGError.png")

    #Calculate RMS:


    print("Average Error: {:.3f} %".format(avg))

def prepare_img(fp, size=(256, 256)):
    im = Image.open(fp)
    im = ImageOps.grayscale(im)
    im = im.resize((256, 256))
    im2 = copy.deepcopy(im)
    im = np.asarray(im)
    im = im[np.newaxis, :, :, np.newaxis]
    return im, im2

def evaluate_on_images(img_path, Model=None, model_path=None):
    images = []
    for f in os.listdir(img_path):
        images.append(f)

    val_ds = tensorflow.keras.utils.image_dataset_from_directory(
        img_path,
        labels=None,
        label_mode="int",
        class_names=None,
        color_mode="grayscale",
        batch_size=1,
        image_size=(256, 256),
        shuffle=False,
        seed=0,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )



    if Model is None:
        Model = tf.keras.models.load_model(model_path)

    outputs = []
    for elem in val_ds:
        out = Model(elem)
        out_pct = float((out + 500)/1000)
        outputs.append(out_pct)

    text = ""
    for i in range(len(outputs)):
        text += images[i] + "," + str(outputs[i]) + "\n"

    fn = os.path.join(img_path, "results.csv")
    with open(fn, "w") as f:
        f.write(text)


if __name__ == "__main__":
    imglbl_path = os.path.join(os.getcwd(), "Width", "Data", "Dataset3", "label.csv")

    #preprocess_img_csv("Dataset3\\label.csv","Dataset3", "label.csv", rename=False)
    #preprocess_img_csv("Dataset2\\label.csv", "Dataset2", "label_test.csv", rename=False)
    #preprocess_img_csv("Test_Dataset\\label.csv", "Test_Dataset", "label_test.csv", rename=True)
    #train_img(imgs=os.path.join("Dataset3", "bild"), label_csv="label.csv", epochs=100)
    #eval_medium_err("Test_Dataset\\label.csv", "Test_Dataset", Model=None, model_path="Models\\model0100.pth", batchsize=10)
    testlbl_ds = os.path.join(os.getcwd(), "Width", "Data", "Test_Dataset", "label.csv")
    test_ds = os.path.join(os.getcwd(), "Width", "Data", "Test_Dataset")
    eval_ds = os.path.join(os.getcwd(), "Width", "Data", "EvaluateReal")
    img_path = os.path.join(os.getcwd(), "Width", "Data", "Dataset3", "bild")
    lbl_path = os.path.join(os.getcwd(), "Width", "label.csv")
    model_pth = os.path.join(os.getcwd(), "Width", "Models", "model0300.pth")
    train_img(imgs=img_path, label_csv=lbl_path, epochs=300, start_lr=1e-3, end_lr=1e-7)
    eval_medium_err(testlbl_ds, test_ds, Model=None, model_path=model_pth,
                    batchsize=BATCHSIZE)
    evaluate_on_images(eval_ds, model_path=model_pth)



