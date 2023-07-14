# import the necessary packages
import os

# initialize the path to the root folder where the dataset resides and the
# path to the train and test directory
DATASET_FOLDER = '/media/rllab/HDD2/nima_bsl/face/Latest-Version/Latest-Version/dataset'
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")

# # CUSTOM_TRAIN_DATASET_FOLDER = f'customDataset/train'
# CUSTOM_TRAIN_DIRECTORY_NEUTRAL = os.path.join(TRAIN_DIRECTORY, "neutral")
# CUSTOM_TRAIN_DIRECTORY_ANGRY = os.path.join(TRAIN_DIRECTORY, "angry")
# CUSTOM_TRAIN_DIRECTORY_FEAR = os.path.join(TRAIN_DIRECTORY, "fear")
# CUSTOM_TRAIN_DIRECTORY_HAPPY = os.path.join(TRAIN_DIRECTORY, "happy")
# CUSTOM_TRAIN_DIRECTORY_SAD = os.path.join(TRAIN_DIRECTORY, "sad")
# CUSTOM_TRAIN_DIRECTORY_SURPRISE = os.path.join(TRAIN_DIRECTORY, "surprise")

# CUSTOM_TEST_DATASET_FOLDER = f'customDataset/test'
# CUSTOM_TEST_DIRECTORY_NEUTRAL = os.path.join(CUSTOM_TEST_DATASET_FOLDER, "neutral")
# CUSTOM_TEST_DIRECTORY_ANGRY = os.path.join(CUSTOM_TEST_DATASET_FOLDER, "angry")
# CUSTOM_TEST_DIRECTORY_FEAR = os.path.join(CUSTOM_TEST_DATASET_FOLDER, "fear")
# CUSTOM_TEST_DIRECTORY_HAPPY = os.path.join(CUSTOM_TEST_DATASET_FOLDER, "happy")
# CUSTOM_TEST_DIRECTORY_SAD = os.path.join(CUSTOM_TEST_DATASET_FOLDER, "sad")
# CUSTOM_TEST_DIRECTORY_SURPRISE = os.path.join(CUSTOM_TEST_DATASET_FOLDER, "surprise")


##train
model_train = "output/model_3_v1.pth"
plot = "output/plot_3_v1.png"


# initialize the amount of samples to use for training and validation
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# specify the batch size, total number of epochs and the learning rate
BATCH_SIZE = 32
NUM_OF_EPOCHS = 150#80#50
LR = 1e-1

##test
prototxt = "model/deploy.prototxt.txt"
caffe_model= "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
model = "output/model_3_v1.pth"
video = "video/1.mp4"
##
cat = "neutral"
image_folder = f"./customDataset/sohrabi/train/{cat}"
output_folder = f"./customDataset/sohrabi/train/{cat}_crop"
dict_path = f"./customDataset/sohrabi/train/{cat}.pkl"

##
confidence = 0.9

####
save_crop = True
show_detection = True
show_emotion = True