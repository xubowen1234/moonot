///|
/// Combine histograms on one fixed support, using either a shared geometry or
/// a distinct caller-supplied ground cost for each input domain.
fn main {
  let histograms = [[0.70, 0.20, 0.10], [0.15, 0.65, 0.20], [0.10, 0.25, 0.65]]
  let cost = [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.15,
    tolerance: 1.0e-8,
    max_iterations: 3000,
  }
  match
    @moonot.regularized_barycenter(histograms, [1.0, 2.0, 1.0], cost, options) {
    Err(_) => println("invalid histogram set or solver configuration")
    Ok(solution) => {
      println("shared-cost weighted barycenter:")
      for i in 0..<solution.barycenter.length() {
        println("bin \{i}: \{solution.barycenter[i]}")
      }
      println("weighted regularized objective:")
      println(solution.weighted_objective)
    }
  }

  let domain_costs = [
    [[0.0, 0.8, 3.0], [1.2, 0.0, 0.7], [4.0, 0.5, 0.0]],
    [[0.0, 2.0, 1.0], [0.4, 0.0, 1.5], [1.2, 0.6, 0.0]],
    [[0.0, 1.4, 2.2], [0.7, 0.0, 0.9], [3.0, 0.3, 0.0]],
  ]
  match
    @moonot.regularized_barycenter_with_costs(
      histograms,
      [1.0, 2.0, 1.0],
      domain_costs,
      options,
    ) {
    Err(_) =>
      println("invalid domain-specific cost set or solver configuration")
    Ok(solution) => {
      println("per-domain-cost weighted barycenter:")
      for i in 0..<solution.barycenter.length() {
        println("bin \{i}: \{solution.barycenter[i]}")
      }
      println("weighted regularized objective:")
      println(solution.weighted_objective)
      println("largest inner transport marginal residual:")
      println(solution.maximum_transport_residual)
    }
  }
}
