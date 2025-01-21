import os

DATA_DIR = 'data/oxford-iiit-pet'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')
LIST_FILE = os.path.join(ANNOTATIONS_DIR, 'list.txt')
TEST_FILE = os.path.join(ANNOTATIONS_DIR, 'test.txt')
MODEL_PATH = "best_pet_classifier.pth"