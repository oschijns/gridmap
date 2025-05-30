//! Cell trait

use num_traits::{ConstZero, Zero};

/// Cell trait
pub trait Cell {
    /// Null value for the empty cells.
    const NULL: Self;

    /// Returns true if the cell is null.
    fn is_null(&self) -> bool;
}

/// Implement Cell trait for all types using num-traits
impl<N> Cell for N
where
    N: Zero + ConstZero,
{
    /// A numeric cell is null if it is zero.
    const NULL: Self = ConstZero::ZERO;

    /// A numeric cell is null if it is zero.
    #[inline]
    fn is_null(&self) -> bool {
        self.is_zero()
    }
}

/// Encapsulation of option type to implement Cell trait
pub struct Opt<T>(pub Option<T>);

/// Implement Cell trait for Option type
impl<T> Cell for Opt<T> {
    /// Null value for the empty cells.
    const NULL: Self = Opt(None);

    /// An option cell is null if it is None.
    #[inline]
    fn is_null(&self) -> bool {
        self.0.is_none()
    }
}
