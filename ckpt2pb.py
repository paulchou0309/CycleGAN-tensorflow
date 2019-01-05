import tensorflow as tf
from tensorflow.python.platform import gfile

meta_path = 'models/cyclegan.model-8002.meta' # Your .meta file

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess, 'models/cyclegan.model-8002')

    # Output nodes
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    
    # for a in output_node_names:
    #     if 'generator' not in a:
    #         print(a)

    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), ['generatorA2B/final_result'])

    # Save the frozen graph
    with gfile.FastGFile('models/graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
