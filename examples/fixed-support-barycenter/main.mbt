///|
/// Combine three discrete histograms on the same support into a weighted
/// entropy-regularized barycenter.
fn main {
  let histograms = [[0.70, 0.20, 0.10], [0.15, 0.65, 0.20], [0.10, 0.25, 0.65]]
  let cost = [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]]
  match
    @moonot.regularized_barycenter(histograms, [1.0, 2.0, 1.0], cost, {
      epsilon: 0.15,
      tolerance: 1.0e-8,
      max_iterations: 3000,
    }) {
    Err(_) => println("invalid histogram set or solver configuration")
    Ok(solution) => {
      println("weighted regularized barycenter:")
      for i in 0..<solution.barycenter.length() {
        println("bin \{i}: \{solution.barycenter[i]}")
      }
      println("weighted regularized objective:")
      println(solution.weighted_objective)
      println("largest inner transport marginal residual:")
      println(solution.maximum_transport_residual)
      println("outer log-barycenter residual:")
      println(solution.log_barycenter_residual)
    }
  }
}
