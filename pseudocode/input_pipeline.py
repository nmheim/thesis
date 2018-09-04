  def normalize(data, vmin=None, vmax=None):
      if vmin is None or vmax is None:
          vmax, vmin = data.max(), data.min()
      return (data-vmin)/np.abs(vmin-vmax)

  def make_generator(sequence_length, xslice, yslice):
      time, array = read_kuro(xslice, yslice)
      array = normalize(array)
      tottime = array.shape[0]
      for i in range(tottime-sequence_length):
          seq = array[i:i+sequence_length]
          t = time[i:i+sequence_length]
          yield t, seq
     
  def pipeline(nr_prev_frames, nr_prev_frames):
      sequence_length = nr_prev_frames + nr_next_frames + 1
      def generator():
          return make_generator(sequence_length, slice(0, 100), slice(0, 100)):
      dataset = tf.data.Dataset.from_generator(
          generator=generator, ...)
      # some transformations like shuffling, batching and resizing ...
      iterator = dataset.make_one_shot_iterator()
      time, series = iterator.get_next()
      inputs, labels = series[:nr_prev_frames], series[nr_prev_frames:]
      features = {'inputs':inputs, 'time':time}
      return features, labels

  if __name__ == '__main__':
      features, labels = pipeline(10, 10)
      inputs = features['inputs']
      with tf.Session() as sess:
          tf.global_variables_initializer().run()
          sess.run(inputs) # ==> retrieves first input series.
          sess.run(inputs) # ==> retrieves second input series.
