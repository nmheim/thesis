  def tikhonov(inputs, labels, states, params):
      """Apply Tikhonov Regularization to weight optimization of Wout:
  
          Wout = Y XT inv(X XT - beta * I), where: X = concat(S, U)
  
      Variables in the equation above if the input is a scalar and we feed T
      timesteps to the network:

          U: concatenated inputs:                   (  1 x T)
          Y: concatenated labels (desired outputs): (  1 x T)
          S: concatenated states:                   (n   x T)
          X: concatenated states and inputs:        (n+1 x T)
          beta: tikhonov reg. parameter             ( scalar)

      Inputs can be set to None. Then only the states are used for calculating
      Wout. By concatenated states/inputs/labels we mean the collected states/
      inputs/labels of T time steps.
      """
      beta = tf.constant(params['tikhonov_beta'], name='beta')
  
      if inputs is None:
          X = tf.identity(states, name="X")
      else:
          X = tf.concat([states, inputs], axis=0, name='X')
      XT = tf.transpose(X, name='XT')
  
      XXT = tf.matmul(X, XT)
      eye = tf.eye(XXT.shape[0].value, dtype=tf.float32)
      inv = tf.linalg.inv(XXT - beta * eye)
      wout = tf.matmul(labels, tf.matmul(XT, inv), name='wout')
      return wout


