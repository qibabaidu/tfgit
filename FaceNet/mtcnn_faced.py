from scipy import misc 
import tensorflow as tf 
import facenet.src.align.detect_face as detect_face
import cv2
import matplotlib.pyplot as plt 
import os

minsize=20
threshold=[0.6, 0.7, 0.7]
factor=0.709
gpu_memory_fraction=0.5

print("Creating networks and loading parameters")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

with tf.Graph().as_default():
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

image_path="/home/ajay/Pros/Python/TFBasic/FaceNet/timg.jpeg"

img = misc.imread(image_path)
bouding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bouding_boxes.shape[0]	# face numbers
print("Face Numsers: {}".format(nrof_faces))
print(bouding_boxes)

crop_faces=[]
for face_position in bouding_boxes:
	face_position = face_position.astype(int)
	print(face_position[0:4])
	cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
	crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2],]
	crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
	print(crop.shape)
	crop_faces.append(crop)
	plt.imshow(crop)
	plt.show()

plt.imshow(img)
plt.show()