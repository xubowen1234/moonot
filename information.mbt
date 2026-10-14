///|
/// Validation and arithmetic errors for dense coupling information diagnostics.
/// Coordinates are zero-based; dimensions are counts, not inferred from mass.
pub(all) enum CouplingInformationError {
  InvalidCoupling(CouplingError)
  InvalidReference(CouplingError)
  RowCountMismatch(Int, Int)
  ColumnCountMismatch(Int, Int)
  ZeroReferenceSupport(Int, Int)
  NumericalOverflow(Int, Int)
}

///|
/// Return `sum(P * (log(P) - 1))`, with zero entries contributing zero.
/// This is the unscaled entropy regularizer, not Shannon entropy. The plan
/// must be non-empty, rectangular, finite and nonnegative; it is not mutated
/// or normalized. Non-finite terms or accumulation return a typed error.
pub fn coupling_entropy(
  coupling : Array[Array[Double]],
) -> Result[Double, CouplingInformationError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(CouplingInformationError::InvalidCoupling(error))
    Ok(value) => value
  }
  let mut total = 0.0
  let mut correction = 0.0
  for i in 0..<coupling.length() {
    for j in 0..<columns {
      let mass = coupling[i][j]
      let term = if mass == 0.0 { 0.0 } else { mass * (@math.ln(mass) - 1.0) }
      match information_accumulate(total, correction, term, i, j) {
        Err(error) => return Err(error)
        Ok((sum, remainder)) => {
          total = sum
          correction = remainder
        }
      }
    }
  }
  Ok(total)
}

///|
/// Return generalized `KL(P || Q) = sum(p * log(p/q) - p + q)`.
/// Both plans must be non-empty rectangular finite nonnegative matrices with
/// identical dimensions; arbitrary and unequal total masses are allowed.
/// At `(0, q)` the contribution is `q` (including `(0, 0) -> 0`);
/// `(p > 0, 0)` returns `ZeroReferenceSupport`, never infinity.
/// Near equality a relative-difference series avoids cancellation; otherwise
/// logarithms are subtracted without constructing the potentially overflowing
/// ratio. Inputs are neither normalized nor mutated.
pub fn coupling_kl(
  coupling : Array[Array[Double]],
  reference : Array[Array[Double]],
) -> Result[Double, CouplingInformationError] {
  let columns = match validate_coupling(coupling) {
    Err(error) => return Err(CouplingInformationError::InvalidCoupling(error))
    Ok(value) => value
  }
  let reference_columns = match validate_coupling(reference) {
    Err(error) => return Err(CouplingInformationError::InvalidReference(error))
    Ok(value) => value
  }
  if coupling.length() != reference.length() {
    return Err(
      CouplingInformationError::RowCountMismatch(
        coupling.length(),
        reference.length(),
      ),
    )
  }
  if columns != reference_columns {
    return Err(
      CouplingInformationError::ColumnCountMismatch(columns, reference_columns),
    )
  }
  let mut total = 0.0
  let mut correction = 0.0
  for i in 0..<coupling.length() {
    for j in 0..<columns {
      let p = coupling[i][j]
      let q = reference[i][j]
      if p > 0.0 && q == 0.0 {
        return Err(CouplingInformationError::ZeroReferenceSupport(i, j))
      }
      let term = if p == q {
        0.0
      } else if p == 0.0 {
        q
      } else {
        information_kl_term(p, q)
      }
      match information_accumulate(total, correction, term, i, j) {
        Err(error) => return Err(error)
        Ok((sum, remainder)) => {
          total = sum
          correction = remainder
        }
      }
    }
  }
  Ok(total)
}

///|
/// Called only for positive, unequal, validated masses.
fn information_kl_term(p : Double, q : Double) -> Double {
  let difference = p - q
  if abs(difference) <= q && abs(difference / q) <= 0.5 {
    // Sterbenz subtraction is exact in this range. With t = (p-q)/q,
    // KL = q*t^2 * sum(n=2..infinity, (-t)^(n-2)/(n*(n-1))).
    // Horner through n=64: at |t|<=1/2 the omitted coefficient tail
    // is below 2^-63/(64*65), well below Double rounding error.
    let t = difference / q
    let mut coefficient = 0.0
    for n = 64; n >= 2; n = n - 1 {
      coefficient = 1.0 / (n.to_double() * (n - 1).to_double()) -
        t * coefficient
    }
    // Scale the relative factor before multiplying the exact difference,
    // rather than rounding q*t*t before applying the coefficient. This
    // retains accuracy when the final result is subnormal.
    difference * (t * coefficient)
  } else {
    let delta = @math.ln(p) - @math.ln(q)
    // Factoring the -p into the product avoids overflowing p*delta
    // when the final divergence still fits in Double. Outside the series
    // region cancellation is bounded, so no negative-value clamp is needed.
    p * (delta - 1.0) + q
  }
}

///|
/// Checked Kahan accumulation shared by both diagnostics.
fn information_accumulate(
  total : Double,
  correction : Double,
  term : Double,
  row : Int,
  column : Int,
) -> Result[(Double, Double), CouplingInformationError] {
  let adjusted = term - correction
  let next = total + adjusted
  let remainder = next - total - adjusted
  if !is_finite(term) ||
    !is_finite(adjusted) ||
    !is_finite(next) ||
    !is_finite(remainder) {
    return Err(CouplingInformationError::NumericalOverflow(row, column))
  }
  Ok((next, remainder))
}
