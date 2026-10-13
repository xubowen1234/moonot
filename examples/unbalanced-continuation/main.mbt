///|
/// Continue an unbalanced distribution-reconciliation solve after changing
/// epsilon and the observed target. Counts are measurements for this input,
/// not a promise that continuation uses fewer iterations.
fn main {
  let source = [0.55, 0.30, 0.15]
  let first_target = [0.20, 0.50, 0.30]
  let next_target = [0.25, 0.45, 0.30]
  let cost = [[0.0, 0.4, 1.1], [0.5, 0.0, 0.3], [1.2, 0.4, 0.0]]
  match
    @moonot.unbalanced_sinkhorn_with_warm_start(
      source,
      first_target,
      cost,
      {
        epsilon: 0.8,
        rho_source: 1.2,
        rho_target: 0.9,
        cost_weight: 1.0,
        tolerance: 1.0e-9,
        max_iterations: 2000,
      },
      None,
    ) {
    Err(_) => println("initial unbalanced solve failed")
    Ok((initial, seed)) => {
      println("initial solve iterations and objective:")
      println(initial.iterations)
      println(initial.objective)
      match
        @moonot.unbalanced_sinkhorn_with_warm_start(
          source,
          next_target,
          cost,
          {
            epsilon: 0.35,
            rho_source: 1.2,
            rho_target: 0.9,
            cost_weight: 1.0,
            tolerance: 1.0e-9,
            max_iterations: 2000,
          },
          Some(seed),
        ) {
        Err(_) => println("continued unbalanced solve failed")
        Ok((continued, _)) => {
          println(
            "continued solve iterations, objective, and scaling residual:",
          )
          println(continued.iterations)
          println(continued.objective)
          println(continued.scaling_residual)
          match
            @moonot.unbalanced_sinkhorn(source, next_target, cost, {
              epsilon: 0.35,
              rho_source: 1.2,
              rho_target: 0.9,
              cost_weight: 1.0,
              tolerance: 1.0e-9,
              max_iterations: 2000,
            }) {
            Err(_) => println("cold comparison solve failed")
            Ok(cold) => {
              println("cold solve iterations and objective:")
              println(cold.iterations)
              println(cold.objective)
            }
          }
        }
      }
    }
  }
}
