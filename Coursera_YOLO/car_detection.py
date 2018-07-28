# importing all the libraries

import keras 
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Model, load_model
from keras.layers import Input , Lambda , Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body #yet another darknet to keras

# The image is of size (608 , 608 ,3) and has 80 classes.
# So , the bounding boxes will be represented by 85 numbers.
# (pc , bx,by,bw,bh) + 80 classes
# Calculating the box details and filtering it out based on the threshold

def bounding_boxes(box_confidence , boxes , box_class_prob , threshold > 0.5):
"""
box_confidence = (grid size , no. of anchor boxes , 1)  ( 19 ,19 ,5 ,1(pc))
boxes = (grid size , no. of anchor boxes , 4) (19,19,5,4(bx,by,bh,bw))
box_class_prob = (grid size , no. of anchor boxes , 80) (19,19,5,no. of classes)
"""
    a = np.random.randn(19*19 ,5,1)
    b = np.random.randn(19*19 ,5,4)
    box_score = a*b
    
    box_classes =keras.backend.argmax(box_score, axis=-1)
    box_class_score = keras.backend.max(box_score, axis=None, keepdims=False)
    
    filtering_mask = box_class_score > threshold
    
    scores = tf.boolean_mask(box_class_score , filtering_mask)
    boxes = tf.boolean_mask(b , filtering_mask)
    classes = tf.boolean_mask(box_classes , filtering_mask)
    
    return scores , boxes , classes

with tf.Session() as test_without_nms:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

# Now , as the next step  , Non-Maximum Supression has to be calculated. 
# Have to find out the area of the boxes , co-ordinates of each box , intersection and the union area.

# calculating the intersection over union 
def iou(box_1 , box_2):
    # co-ordinates of the two boxes 
    x1 = max(box_1[0] , box_2[0])
    y1 = max(box_1[1] , box_2[1])
    x2 = min(box_2[2] , box_2[2])
    y2 = min(box_2[3] , box_2[3])
    intersection_area = (y2-y1)*(x2-x1)
    box_1_area = (box_1[3]-box_1[1])*(box_1[2]-box_1[0])
    box_2_area = (box_2[3]-box_2[1])*(box_2[2]-box_2[0])
    union_area = box_1_area + box_2_area - intersection_area
    iou = intersection_area / union_area
    
    print('The value of intersection-over-union is : ',iou)
    print('box_1 co-ordinates :' , box_1)
    print('box_2 co-ordinates :' , box_2)
    
    return iou

# Implementing Non-max-suppression
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 

    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold=iou_threshold) #tf has a built-in function for non-max-suppression.
    # so there is no need to implement it separately.
    
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
   
    
    return scores, boxes, classes
    
# testing the model after implementing the nms to ake sure that there is a improvement in the score

with tf.Session() as test_with_nms:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
    
# Evaluation

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs,score_threshold)
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes ,max_boxes,iou_threshold)
    
    return scores, boxes, classes
    
with tf.Session() as test:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
    
# Test the YOLO model on images(pre-trained)

sess = K.get_session() # start a session

# define classes , anchors and image shape
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.) 

# loading a pre-trained model
yolo_model = load_model("model_data/yolo.h5")

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores,boxes,classes],feed_dict={yolo_model.input:image_data,K.learning_phase():0})
     print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes
    
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

