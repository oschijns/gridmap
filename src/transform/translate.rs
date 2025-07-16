//! Translate indexes

use super::Transform;

/// Translate the index
#[derive(Debug)]
pub struct Translate<const D: usize>(pub [isize; D]);

/// Create a null translation
impl<const D: usize> Default for Translate<D> {
    fn default() -> Self {
        Self([0; D])
    }
}

/// Apply the transformation to the given index
impl<const D: usize> Transform<D> for Translate<D> {
    fn apply(&self, index: &mut [isize; D]) {
        // flip the indexes
        for (d, i) in index.iter_mut().enumerate() {
            *i += self.0[d];
        }
    }
}
