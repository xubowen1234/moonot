///|
/// Run a small balanced Sinkhorn smoke test using the public MoonOT API.
fn main {
  let source = [0.5, 0.5]
  let target = [0.5, 0.5]
  let cost = [[0.0, 1.0], [1.0, 0.0]]
  let options : @moonot.SinkhornOptions = {
    epsilon: 0.5,
    tolerance: 1.0e-9,
    max_iterations: 2000,
  }

  println("MoonOT balanced Sinkhorn smoke test")
  println("Input source masses:")
  for i in 0..<source.length() {
    println("  a[\{i}] = \{source[i]}")
  }
  println("Input target masses:")
  for j in 0..<target.length() {
    println("  b[\{j}] = \{target[j]}")
  }
  println("Input cost matrix:")
  for i in 0..<cost.length() {
    for j in 0..<cost[i].length() {
      println("  C[\{i}, \{j}] = \{cost[i][j]}")
    }
  }
  println(
    "Options: epsilon=\{options.epsilon}, tolerance=\{options.tolerance}, max_iterations=\{options.max_iterations}",
  )

  match @moonot.balanced_sinkhorn(source, target, cost, options) {
    Err(error) => report_solver_error(error)
    Ok(solution) => {
      println("Coupling:")
      for i in 0..<solution.coupling.length() {
        for j in 0..<solution.coupling[i].length() {
          println("  P[\{i}, \{j}] = \{solution.coupling[i][j]}")
        }
      }
      println("Regularized objective: \{solution.regularized_objective}")
      println("Maximum marginal residual: \{solution.marginal_residual}")
      println("Iterations: \{solution.iterations}")
      match solution.termination {
        @moonot.SinkhornTermination::Converged =>
          println("Termination: converged")
        @moonot.SinkhornTermination::IterationLimit =>
          println(
            "Termination: iteration limit reached; requested tolerance was not met",
          )
      }
    }
  }
}

///|
fn report_solver_error(error : @moonot.SinkhornError) -> Unit {
  match error {
    @moonot.SinkhornError::EmptySourceMarginal =>
      println("ERROR: source marginal must not be empty")
    @moonot.SinkhornError::EmptyTargetMarginal =>
      println("ERROR: target marginal must not be empty")
    @moonot.SinkhornError::CostRowCountMismatch =>
      println(
        "ERROR: cost matrix row count must match the source marginal length",
      )
    @moonot.SinkhornError::CostColumnCountMismatch(row) =>
      println("ERROR: cost matrix row \{row} has the wrong number of columns")
    @moonot.SinkhornError::NegativeSourceMass(index) =>
      println("ERROR: source mass at index \{index} is negative")
    @moonot.SinkhornError::NegativeTargetMass(index) =>
      println("ERROR: target mass at index \{index} is negative")
    @moonot.SinkhornError::NonFiniteSourceMass(index) =>
      println("ERROR: source mass at index \{index} is not finite")
    @moonot.SinkhornError::NonFiniteTargetMass(index) =>
      println("ERROR: target mass at index \{index} is not finite")
    @moonot.SinkhornError::NonFiniteTotalMass =>
      println("ERROR: total marginal mass is not finite")
    @moonot.SinkhornError::UnequalTotalMass =>
      println(
        "ERROR: balanced transport requires equal source and target total mass",
      )
    @moonot.SinkhornError::NegativeCost(row, column) =>
      println("ERROR: cost at (\{row}, \{column}) is negative")
    @moonot.SinkhornError::NonFiniteCost(row, column) =>
      println("ERROR: cost at (\{row}, \{column}) is not finite")
    @moonot.SinkhornError::InvalidEpsilon =>
      println("ERROR: epsilon must be finite and positive")
    @moonot.SinkhornError::InvalidTolerance =>
      println("ERROR: tolerance must be finite and positive")
    @moonot.SinkhornError::InvalidIterationLimit =>
      println("ERROR: maximum iterations must be positive")
    @moonot.SinkhornError::InvalidWarmStart =>
      println("ERROR: warm-start potential dimensions or values are invalid")
    @moonot.SinkhornError::NumericalFailure =>
      println("ERROR: the solver encountered a numerical failure")
  }
}
