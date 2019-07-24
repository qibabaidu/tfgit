import os
import numpy as np 
from tensorflow.python import pywrap_tensorflow

checkpoint_path = "/home/ajay/Pros/Python/TFBasic/ResNetCifar10/train_dir/model.ckpt-54189"
print checkpoint_path

# read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

data_print=np.array([])
for key in var_to_shape_map:
	print "tensor_name: ", key
	ckpt_data = np.array(reader.get_tensor(key))
	ckpt_data = ckpt_data.flatten()
	data_print = np.append(data_print, ckpt_data, axis=0)

print(data_print,data_print.shape,np.max(data_print),np.min(data_print),np.mean(data_print))
