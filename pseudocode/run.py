   #!/usr/bin/env python
   import tensorflow as tf
   from torsk.models import lms
   import torsk.datasets as ds
   from torsk.esn import ESNCell
   from torks.visualize import animate_double_imshow
   
   params = {
       "esn_cell": ESNCell,
       "esn_spectral_radius": 1.5,
       "activation": tf.tanh,
       # ... some more setup parameters
   }
   
   with tf.variable_scope("input_pipeline"):
       features, labels = ds.train_mackey_input_fn(
           params["nr_prev_frames"], params["nr_next_frames"], shuffle=False)
       inputs = features['inputs']
   
   lms.build_model(inputs, labels, params)
   
   states = tf.get_default_graph().get_tensor_by_name("encoder/all_states:0")
   with tf.variable_scope("summaries/"):
       tf.summary.tensor_summary('encoder_states', states)
       tf.summary.merge_all()
   
   fetches = {"prediction": "decoder/prediction:0", ...}
   with tf.Session() as sess:
       sess.run("optimize/optimize:0")
       leaves = sess.run(fetches)

   animate_double_imshow(leaves['prediction'], leavs['target'])
