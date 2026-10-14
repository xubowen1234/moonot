///|
/// Validation errors from coupling reorientation; permutation indices are zero-based.
pub(all) enum CouplingOrientationError {
  InvalidCoupling(CouplingError)
  SourcePermutationLength
  TargetPermutationLength
  SourceIndexOutOfRange(Int)
  TargetIndexOutOfRange(Int)
  DuplicateSourceIndex(Int)
  DuplicateTargetIndex(Int)
}

///|
/// Return a fresh target-by-source transpose. No storage is shared with input.
pub fn coupling_transpose(
  coupling : Array[Array[Double]],
) -> Result[Array[Array[Double]], CouplingError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(error)
    Ok(value) => value
  }
  let result : Array[Array[Double]] = []
  for j in 0..<columns {
    let row = Array::make(coupling.length(), 0.0)
    for i in 0..<coupling.length() {
      row[i] = coupling[i][j]
    }
    result.push(row)
  }
  Ok(result)
}

///|
/// Reindex both supports without losing mass: output[i][j] = input[source[i]][target[j]].
/// Each index array must be a complete bijection of its respective support.
/// Inputs are validated before any output is constructed, and are not modified.
pub fn coupling_permute(
  coupling : Array[Array[Double]],
  source_order : Array[Int],
  target_order : Array[Int],
) -> Result[Array[Array[Double]], CouplingOrientationError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(CouplingOrientationError::InvalidCoupling(error))
    Ok(value) => value
  }
  if source_order.length() != coupling.length() {
    return Err(CouplingOrientationError::SourcePermutationLength)
  }
  if target_order.length() != columns {
    return Err(CouplingOrientationError::TargetPermutationLength)
  }
  let seen_source = Array::make(coupling.length(), false)
  for i in 0..<source_order.length() {
    let index = source_order[i]
    if index < 0 || index >= coupling.length() {
      return Err(CouplingOrientationError::SourceIndexOutOfRange(i))
    }
    if seen_source[index] {
      return Err(CouplingOrientationError::DuplicateSourceIndex(i))
    }
    seen_source[index] = true
  }
  let seen_target = Array::make(columns, false)
  for j in 0..<target_order.length() {
    let index = target_order[j]
    if index < 0 || index >= columns {
      return Err(CouplingOrientationError::TargetIndexOutOfRange(j))
    }
    if seen_target[index] {
      return Err(CouplingOrientationError::DuplicateTargetIndex(j))
    }
    seen_target[index] = true
  }
  let result : Array[Array[Double]] = []
  for i in 0..<source_order.length() {
    let row = Array::make(columns, 0.0)
    for j in 0..<columns {
      row[j] = coupling[source_order[i]][target_order[j]]
    }
    result.push(row)
  }
  Ok(result)
}
