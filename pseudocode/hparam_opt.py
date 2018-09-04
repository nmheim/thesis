  import skopt
  from skopt.space import Real

  opt_steps = 100
  output_dir = "hpopt"
  dimensions = [
      Real(low=1e-5, high=1.0, name="esn_sparsity", prior="log_uniform"),
      # some more of those ...
  ]
  initial_values = [
      0.1,    # esn_sparsity
      # ...
  ]
  params = {
      # defines all default network parameters
  }
  
  @skopt.utils.use_named_args(dimensions=dimensions)
  def fitness(sampled_params):
      params.update(sampled_params)
  
      features, labels = ds.input_fn(
          params["nr_prev_frames"],
          params["nr_next_frames"],
          shuffle=True)
      inputs = features['inputs']
      
      lms.build_model(inputs, labels, params)
      leaves = lms.train(params)
      # make 10 predictions and calculate their mean prediction error as a 'metric'
  
      if not np.isfinite(metric):
          metric = 1e10
      return metric
  
  if __name__ == "__main__":
  
      search_result = skopt.gp_minimize(
          func=fitness, dimensions=dimensions, acq_func="gp_hedge",
          n_calls=opt_steps, x0=initial_values)
      
      skopt.dump(
          search_result, os.path.join(output_dir, "result.pkl"),
          store_objective=False)
