import os
import random


"""
read all image names from the folder and split them into train and test
"""
def split_data(folder_path, train_path, test_path, split_ratio=0.8):
    # get all image names
    images_dir = os.path.join(folder_path, 'images')
    masks_dir = os.path.join(folder_path, 'masks')
    image_names = os.listdir(images_dir)
    image_names = [image_name for image_name in image_names if image_name.endswith('.jpg') or image_name.endswith('.png')]
    # randomly split the image names
    random.seed(0)
    random.shuffle(image_names)
    split_index = int(len(image_names) * split_ratio)
    train_image_names = image_names[:split_index]
    test_image_names = image_names[split_index:]
    
    # write the image names to the train and test files
    with open(train_path, 'w') as f:
        for image_name in train_image_names:
            mask_name = image_name[:-4] + '.png'
            f.write(f"images/{image_name} masks/{mask_name}\n")
    with open(test_path, 'w') as f:
        for image_name in test_image_names:
            mask_name = image_name[:-4] + '.png'
            f.write(f"images/{image_name} masks/{mask_name}\n")
   
if __name__ == '__main__':
    folder_path = 'data/merge'
    train_path = 'train.txt'
    test_path = 'test.txt'
    split_data(folder_path, train_path, test_path, split_ratio=0.8)