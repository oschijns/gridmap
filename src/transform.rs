//! Transform indexes given some parameters

/// Mirror the index
pub mod mirror;

/// Rotate the index
pub mod rotate;

/// Translate the index
pub mod translate;

/// Define a set of parameters to transform indexes
pub trait Transform<const D: usize> {
    /// Apply a transformation to an index
    fn apply(&self, index: &mut [isize; D]);

    /// Apply trasnformation to an index
    fn transform(&self, index: &[isize; D]) -> [isize; D] {
        let mut new_index = *index;
        self.apply(&mut new_index);
        new_index
    }
}

/// Apply a sequence of transformations to an index
impl<const D: usize> Transform<D> for &[&dyn Transform<D>] {
    /// Apply a sequence of transformations to an index
    fn apply(&self, index: &mut [isize; D]) {
        for trs in self.iter() {
            trs.apply(index);
        }
    }
}
