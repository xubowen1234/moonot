///|
/// Reconcile distributions with unmatched mass, such as missing or newly
/// observed categories. The separate penalties control how strongly each
/// side's observed marginal should be preserved.
fn main {
  let source = [0.45, 0.35, 0.20]
  let target = [0.25, 0.30, 0.35, 0.20]
  let cost = [[0.0, 0.4, 1.2, 1.8], [0.4, 0.0, 0.5, 1.1], [1.2, 0.5, 0.0, 0.4]]
  match
    @moonot.unbalanced_sinkhorn(source, target, cost, {
      epsilon: 0.15,
      rho_source: 2.0,
      rho_target: 0.8,
      cost_weight: 1.0,
      tolerance: 1.0e-9,
      max_iterations: 2000,
    }) {
    Err(_) => println("invalid distributions or solver configuration")
    Ok(solution) => {
      println("divisible transport plan:")
      for i in 0..<solution.coupling.length() {
        for j in 0..<solution.coupling[i].length() {
          println("P[\{i},\{j}] = \{solution.coupling[i][j]}")
        }
      }
      println("objective:")
      println(solution.objective)
      println("transport cost and KL components:")
      println(solution.transport_cost)
      println(solution.reference_kl)
      println(solution.source_marginal_kl)
      println(solution.target_marginal_kl)
      println("marginal residuals (diagnostic, not balanced constraints):")
      println(solution.source_marginal_residual)
      println(solution.target_marginal_residual)
    }
  }
}
