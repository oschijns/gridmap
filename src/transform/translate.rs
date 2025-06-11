//! Translate indexes

use super::Transform;

/// Translate the index
pub struct Translate<const D: usize>(pub [isize; D]);

/// Apply the transformation to the given index
impl<const D: usize> Transform<D> for Translate<D> {
    fn apply(&self, index: &mut [isize; D]) {
        // flip the indexes
        for (d, i) in index.iter_mut().enumerate() {
            *i += self.0[d];
        }
    }
}
