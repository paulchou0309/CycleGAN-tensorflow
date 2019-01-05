# import tensorflow as tf

# from tensorflow.python.platform import gfile
# model_path = "models/graph.pb"

# # read graph definition
# f = gfile.FastGFile(model_path, "rb")
# gd = graph_def = tf.GraphDef()
# graph_def.ParseFromString(f.read())

# # fix nodes
# for node in graph_def.node:
#     print(node.name)
#     if node.op == 'RefSwitch':
#         node.op = 'Switch'
#         for index in xrange(len(node.input)):
#             if 'moving_' in node.input[index]:
#                 node.input[index] = node.input[index] + '/read'
#     elif node.op == 'AssignSub':
#         node.op = 'Sub'
#         if 'use_locking' in node.attr:
#             del node.attr['use_locking']
#     elif node.op == 'AssignAdd':
#         node.op = 'Add'
#         if 'use_locking' in node.attr:
#             del node.attr['use_locking']
#     elif node.op == 'Assign':
#         node.op = 'Identity'
#         if 'use_locking' in node.attr:
#             del node.attr['use_locking']
#         if 'validate_shape' in node.attr:
#             del node.attr['validate_shape']
#         if len(node.input) == 2:
#             # input0: ref: Should be from a Variable node. May be uninitialized.
#             # input1: value: The value to be assigned to the variable.
#             node.input[0] = node.input[1]
#             del node.input[1]

# # import graph into session
# tf.import_graph_def(graph_def, name='')
# tf.train.write_graph(graph_def, './', 'models/graph.pb', as_text=False)
# tf.train.write_graph(graph_def, './', 'good_frozen.pbtxt', as_text=True)

import numpy as np
import tensorflow as tf
import scipy.misc
# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="/Users/tpz/Desktop/CycleGAN-tensorflow/models/graph.lite")
interpreter.allocate_tensors()


try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread
fine_size = 256
path = './datasets/photo2anime/testA/1.jpg'
img_A = _imread(path, mode='RGB').astype(np.float)
img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])

sample_image=np.array([img_A]).astype(np.float32)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
# input_shape = input_details[0]['shape']
# change the following line to feed into your own data.
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_image)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
scipy.misc.imsave('test.jpg', output_data[0])
print(output_data)
