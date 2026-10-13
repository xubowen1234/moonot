///|
/// Construct geometric costs and solve a small point-cloud transport problem.
fn main {
  let source_points = [[0.0], [2.0]]
  let target_points = [[0.0], [2.0]]
  match @moonot.squared_euclidean_cost_matrix(source_points, target_points) {
    Err(_) => println("invalid point-cloud inputs")
    Ok(cost) =>
      match
        @moonot.balanced_sinkhorn([0.4, 0.6], [0.2, 0.8], cost, {
          epsilon: 0.2,
          tolerance: 1.0e-9,
          max_iterations: 2000,
        }) {
        Err(_) => println("transport solve did not accept the inputs")
        Ok(solution) => {
          println("point-cloud squared-distance cost matrix:")
          for i in 0..<cost.length() {
            for j in 0..<cost[i].length() {
              println("C[\{i},\{j}] = \{cost[i][j]}")
            }
          }
          println("regularized coupling:")
          for i in 0..<solution.coupling.length() {
            for j in 0..<solution.coupling[i].length() {
              println("P[\{i},\{j}] = \{solution.coupling[i][j]}")
            }
          }
          println("maximum marginal residual:")
          println(solution.marginal_residual)
        }
      }
  }
}
