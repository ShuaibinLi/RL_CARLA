import numpy as np
import cv2
import torch
import glob as glob
from faster_rcnn_model import create_model
import matplotlib.pyplot as plt

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=5).to(device)
model.load_state_dict(torch.load(
    'outputs/model10.pth', map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = '/media/karthikragunath/Personal-Data/carla_6/RL_CARLA/data/test_data'
test_images = glob.glob("{DIR_TEST}/*".format(DIR_TEST=DIR_TEST))
print("Test instances:", len(test_images))
# classes: 0 index is reserved for background
CLASSES = [
    '0', '1', '2', '3', '4'
]

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8

'''
concatenated_images = torch.tensor([])

for i in range(0, 2, 1):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    print('#' * 50, "Image Shape:", image.shape, '#' * 50)
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    if concatenated_images.numel() == 0:
        concatenated_images = image
    else:
        concatenated_images = torch.concat((concatenated_images, image), dim=0)

with torch.no_grad():
    outputs = model(concatenated_images)

print(outputs)
''''''
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            transform = A.Compose(
                [
                    Crop(x_min=int(box[0]), y_min=int(box[1]), x_max=int(box[2]), y_max=int(box[3]))
                ])
            transformed_image = transform(image = orig_image)

            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            sampled_image = torch.tensor(transformed_image['image'])
            sampled_image = cv2.cvtColor(np.float32(sampled_image), cv2.COLOR_RGB2GRAY)
            fig = plt.figure()
            plt.imshow(sampled_image)
            plt.savefig('data/cropped_test_images/' + 'testimage' + '_{image_index}_{box_index}.png'.format(image_index=i, box_index=j))
            # cv2.putText(orig_image, pred_classes[j], 
            #             (int(box[0]), int(box[1]-5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
            #             2, lineType=cv2.LINE_AA)
        plt.imshow(orig_image)
        plt.savefig("test_prediction" + str(i) + ".png")
    print(f"Image {i+1} done...")
    print('-'*50)
''''''
print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()
'''

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            # cv2.putText(orig_image, pred_classes[j],
            #             (int(box[0]), int(box[1]-5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
            #             2, lineType=cv2.LINE_AA)
        # cv2.imshow('Prediction', orig_image)
        # cv2.waitKey(1)
        # cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image,)
        plt.imshow(orig_image)
        plt.savefig("output_images/test_prediction_" + str(i) + ".png")
    print("Image {image_num} done...".format(image_num=(i + 1)))
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')