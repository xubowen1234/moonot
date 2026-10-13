///|
/// Build a soft color-bin mapping from two normalized histograms.
/// Pixel decoding and color-space conversion belong to the caller.
fn main {
  let source_histogram = [0.6, 0.3, 0.1]
  let target_histogram = [0.2, 0.5, 0.3]
  let palette_cost = [[0.0, 0.5, 2.0], [0.5, 0.0, 0.5], [2.0, 0.5, 0.0]]
  match
    @moonot.balanced_sinkhorn(
      source_histogram,
      target_histogram,
      palette_cost,
      { epsilon: 0.2, tolerance: 1.0e-9, max_iterations: 3000, },
    ) {
    Err(_) => println("invalid histograms or solver configuration")
    Ok(solution) => {
      println("transport mass between palette bins:")
      for i in 0..<solution.coupling.length() {
        for j in 0..<solution.coupling[i].length() {
          println("P[\{i},\{j}] = \{solution.coupling[i][j]}")
        }
      }
      println(
        "use each row, normalized by its source mass, as a soft palette map",
      )
      println(solution.marginal_residual)
    }
  }
}
