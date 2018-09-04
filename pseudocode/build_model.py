  def prediction_helper(initial_inputs, params):
      def initialize_fn():
          finished = tf.tile([False], [1,])
          next_inputs = initial_inputs
          return finished, next_inputs
  
      def sample_fn(time, outputs, state):
          """unnecessary for our task..."""
          return tf.constant([0])
  
      def next_inputs_fn(time, outputs, state, sample_ids):
          """Creates next inputs to ESNCell.call(inputs, state)
          Params:
              time: current time step
              outputs: previous cell output. here: (prediction, state)
              state: previous cell state
              sample_ids: unused
          """
          finished = time >= params["nr_next_frames"] - 1
          next_inputs, _ = outputs # pick only prediction, discard state
          return finished, next_inputs, state
  
      helper = tf.contrib.seq2seq.CustomHelper(
          initialize_fn=initialize_fn,
          sample_fn=sample_fn,
          next_inputs_fn=next_inputs_fn)
      return helper

  def build_decoder(init_dec_input, init_dec_state, params)
      with tf.variable_scope("decoder"):
          dec_helper = prediction_helper(
              init_dec_input, init_dec_output, params)
          decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=cell, helper=dec_helper, initial_state=init_dec_state)
          dec_output, dec_state, _ = tf.contrib.seq2seq.dynamic_decode(
              decoder=decoder, output_time_major=None)
      dec_output, dec_states = dec_output.rnn_output
      return dec_output, dec_states
