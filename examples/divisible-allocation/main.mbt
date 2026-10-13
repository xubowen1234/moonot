///|
/// Allocate divisible supplies to demands at minimum regularized cost.
/// This balanced example uses equal total supply and demand; it is not an
/// integer assignment solver and does not model unmet demand.
fn main {
  let supply = [8.0, 5.0]
  let demand = [6.0, 7.0]
  let cost = [[1.0, 3.0], [2.0, 1.0]]
  match
    @moonot.balanced_sinkhorn(supply, demand, cost, {
      epsilon: 0.5,
      tolerance: 1.0e-8,
      max_iterations: 2000,
    }) {
    Err(_) => println("invalid supplies, demands, or solver configuration")
    Ok(solution) => {
      println("divisible allocation plan (supply rows × demand columns):")
      for i in 0..<solution.coupling.length() {
        for j in 0..<solution.coupling[i].length() {
          println("P[\{i},\{j}] = \{solution.coupling[i][j]}")
        }
      }
      println("objective and marginal residual:")
      println(solution.regularized_objective)
      println(solution.marginal_residual)
    }
  }
}
