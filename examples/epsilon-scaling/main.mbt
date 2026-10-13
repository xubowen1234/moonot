///|
/// Continue a balanced solve from a coarse epsilon to a finer epsilon while
/// slightly changing the target histogram.
fn main {
  let source = [0.31, 0.28, 0.41]
  let initial_target = [0.11, 0.56, 0.33]
  let target = [0.11, 0.53, 0.36]
  let cost = [[5.13, 0.15, 4.65], [2.24, 3.79, 0.89], [6.44, 4.28, 2.27]]
  match
    @moonot.balanced_sinkhorn_with_warm_start(
      source,
      initial_target,
      cost,
      { epsilon: 2.0, tolerance: 1.0e-9, max_iterations: 1000, },
      None,
    ) {
    Err(_) => println("transport solve failed")
    Ok((coarse, seed)) => {
      println("coarse epsilon iterations:")
      println(coarse.iterations)
      match
        @moonot.balanced_sinkhorn_with_warm_start(
          source,
          target,
          cost,
          { epsilon: 0.5, tolerance: 1.0e-9, max_iterations: 1000, },
          Some(seed),
        ) {
        Err(_) => println("transport solve failed")
        Ok((fine, _)) => {
          println("fine epsilon iterations with warm start:")
          println(fine.iterations)
          match
            @moonot.balanced_sinkhorn(source, target, cost, {
              epsilon: 0.5,
              tolerance: 1.0e-9,
              max_iterations: 1000,
            }) {
            Err(_) => println("transport solve failed")
            Ok(cold) => {
              println("fine epsilon iterations from a cold start:")
              println(cold.iterations)
            }
          }
        }
      }
    }
  }
}
