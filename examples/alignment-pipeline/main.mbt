///|
/// Align three finite feature distributions through a shared bridge support.
/// Every failed validation or convergence check terminates this workflow.
fn main {
  let source_points = [[0.0, 0.1], [1.0, 0.8]]
  let bridge_points = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
  let target_points = [[0.1, 0.0], [0.9, 1.0]]
  let source = [0.4, 0.6]
  let bridge = [0.25, 0.5, 0.25]
  let target = [0.55, 0.45]
  let weights = [1.0, 0.3]
  let left_cost = match
    @moonot.weighted_squared_euclidean_cost_matrix(
      source_points, bridge_points, weights,
    ) {
    Ok(value) => value
    Err(_) => abort("invalid source-to-bridge geometry")
  }
  let right_cost = match
    @moonot.weighted_squared_euclidean_cost_matrix(
      bridge_points, target_points, weights,
    ) {
    Ok(value) => value
    Err(_) => abort("invalid bridge-to-target geometry")
  }
  let schedule : Array[@moonot.SinkhornOptions] = [
    { epsilon: 1.0, tolerance: 1.0e-10, max_iterations: 2000, },
    { epsilon: 0.2, tolerance: 1.0e-10, max_iterations: 2000, },
  ]
  let left = match
    @moonot.balanced_sinkhorn_continuation(
      source,
      bridge,
      left_cost,
      schedule,
      None,
    ) {
    Ok(value) => value
    Err(_) => abort("source-to-bridge solve failed")
  }
  let right = match
    @moonot.balanced_sinkhorn_continuation(
      bridge,
      target,
      right_cost,
      schedule,
      None,
    ) {
    Ok(value) => value
    Err(_) => abort("bridge-to-target solve failed")
  }
  match (left.termination, right.termination) {
    (@moonot.Converged, @moonot.Converged) => ()
    _ => abort("iteration budget exhausted")
  }
  let left_plan = match
    @moonot.round_balanced_coupling(
      left.stages[1].result.coupling,
      source,
      bridge,
      1.0e-12,
    ) {
    Ok(value) => value.coupling
    Err(_) => abort("left feasibility repair failed")
  }
  let right_plan = match
    @moonot.round_balanced_coupling(
      right.stages[1].result.coupling,
      bridge,
      target,
      1.0e-12,
    ) {
    Ok(value) => value.coupling
    Err(_) => abort("right feasibility repair failed")
  }
  let composed = match
    @moonot.coupling_compose(left_plan, right_plan, 2.0e-12) {
    Ok(value) => value
    Err(_) => abort("incompatible bridge marginals")
  }
  let marginals = match @moonot.coupling_marginals(composed) {
    Ok(value) => value
    Err(_) => abort("invalid composed plan")
  }
  for i in 0..<source.length() {
    if (marginals.source[i] - source[i]).abs() > 3.0e-12 {
      abort("composed source mass not preserved")
    }
  }
  for j in 0..<target.length() {
    if (marginals.target[j] - target[j]).abs() > 3.0e-12 {
      abort("composed target mass not preserved")
    }
  }
  let projected = match
    @moonot.coupling_barycentric_projection(composed, target_points) {
    Ok(value) => value
    Err(_) => abort("conditional feature projection failed")
  }
  for row in projected {
    if row[0] < 0.1 - 1.0e-12 ||
      row[0] > 0.9 + 1.0e-12 ||
      row[1] < 0.0 ||
      row[1] > 1.0 + 1.0e-12 {
      abort("projection left target convex hull")
    }
  }
  println("source -> bridge -> target coupling:")
  for row in composed {
    for mass in row {
      println(mass)
    }
  }
  println("source features mapped into target coordinates:")
  for point in projected {
    for coordinate in point {
      println(coordinate)
    }
  }
  println("stage iterations (left, right):")
  println(left.total_iterations)
  println(right.total_iterations)
}
