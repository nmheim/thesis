  def sliding_score(errors, small_window, large_window, nr_prev_frames):
      index = tf.constant(0)
      shape = [errors.shape[0].value - nr_prev_frames] + errors.shape[1:]
      scores = tf.zeros(shape)
  
      def cond(i, errors, scores):
          return i < shape[0]
  
      def body(i, errors, scores):
          # index at ith prediction (i+nr_prev_frames'th label)
          j = i + nr_prev_frames
          small_errors = errors[j:j+small_window]
          large_errors = errors[j-large_window:j+small_window]
  
          mu, var = tf.nn.moments(large_errors, axes=[0])
          small_mu = tf.reduce_mean(small_errors, axis=0)
  
          x = tf.abs(mu - small_mu) / tf.sqrt(var)
          s = qfunction(x)
  
          scores = tf.concat([scores[:i], [s], scores[i+1:]], axis=0)
          scores.set_shape(shape)
          return i+1, errors, scores
  
      _, _, scores = tf.while_loop(cond, body, [index, errors, scores])
      return tf.identity(scores, name='score')
