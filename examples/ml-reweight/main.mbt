///|
/// Reweight a three-bin empirical distribution toward a target prior.
/// A downstream estimator can consume each source bin's mass in the plan.
fn main {
  match
    @moonot.balanced_sinkhorn(
      [0.2, 0.5, 0.3],
      [0.4, 0.4, 0.2],
      [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]],
      { epsilon: 0.4, tolerance: 1.0e-8, max_iterations: 2000, },
    ) {
    Err(_) => println("invalid distribution or solver configuration")
    Ok(solution) => {
      println("coupling rows correspond to source bins:")
      for i in 0..<solution.coupling.length() {
        for j in 0..<solution.coupling[i].length() {
          println("P[\{i},\{j}] = \{solution.coupling[i][j]}")
        }
      }
      println("regularized objective:")
      println(solution.regularized_objective)
      println("marginal residual:")
      println(solution.marginal_residual)
    }
  }
}
