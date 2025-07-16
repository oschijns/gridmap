//! Mirror indexes

use super::Transform;

/// Mirror the index
#[derive(Debug)]
pub struct Mirror<const D: usize>(pub [bool; D]);

/// Create a null translation
impl<const D: usize> Default for Mirror<D> {
    fn default() -> Self {
        Self([false; D])
    }
}

/// Apply the transformation to the given index
impl<const D: usize> Transform<D> for Mirror<D> {
    fn apply(&self, index: &mut [isize; D]) {
        // flip the indexes
        for (d, i) in index.iter_mut().enumerate() {
            *i = if self.0[d] { -*i } else { *i };
        }
    }
}
