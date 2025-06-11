//! Mirror indexes

use super::Transform;

/// Mirror the index
pub struct Mirror<const D: usize>(pub [bool; D]);

/// Apply the transformation to the given index
impl<const D: usize> Transform<D> for Mirror<D> {
    fn apply(&self, index: &mut [isize; D]) {
        // flip the indexes
        for (d, i) in index.iter_mut().enumerate() {
            *i = if self.0[d] { -*i } else { *i };
        }
    }
}
