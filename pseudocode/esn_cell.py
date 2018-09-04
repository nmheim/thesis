  class ESNCell(LayerRNNCell):
      def __init__(self, num_units, **kwargs):
          super(ESNCell, self).__init__(trainable=False, name=name, **kwargs)
          self._num_units = num_units
          self._activation = activation or math_ops.tanh
          self._spectral_radius = spectral_radius or 0.95
          self._density = density or 0.01
          # define initializers that are called in self.build
   
      @property
      def state_size(self):
          return self._num_units
  
      @property
      def output_size(self):
          return (self._out_units, self._num_units)
  
      def build(self, inputs_shape):
          self._reservoir = self.add_variable(
              name="reservoir", shape=[self._num_units, self._num_units],
              initializer=self._reservoir_init,
              trainable=False)
          # initialize all other weights ...
          self.built = True
  
      def call(self, inputs, state):
  
          x_input = math_ops.matmul(inputs, self._input_weights, name="Win_u")
          x_state = math_ops.matmul(state, self._reservoir, name="W_x")
          new_state = self._activation(x_input + x_state + self._input_biases)
  
          ext_state = array_ops.concat([new_state, inputs], axis=1)
          output = math_ops.matmul(ext_state, self._output_weights)
          return (output, new_state), new_state
