//! Bounding box module

use crate::{cell::Cell, gridmap::GridMap, transform::Transform};
use core::{hash::Hash, ops::IndexMut};
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::{AsPrimitive, ConstZero};

/// Compute the boundaries of the gridmap
pub mod boundaries;

/// Boundaries
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct BoundingBox<const D: usize> {
    /// starting point of the box
    pub start: [isize; D],

    /// ending point of the box
    pub end: [isize; D],
}

/// Define the default bounding as covering the whole space
impl<const D: usize> Default for BoundingBox<D> {
    fn default() -> Self {
        Self {
            start: [isize::MIN; D],
            end: [isize::MAX; D],
        }
    }
}

impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    /// Copy a portion of the source gridmap to the target gridmap with the given transformation
    pub fn copy_to(&self, target: &mut Self, transforms: &[&dyn Transform<D>])
    where
        A: Default + Copy,
        Ic: Eq + Hash + ConstZero + From<isize> + AsPrimitive<isize>,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
    {
        // Transform the indexes and apply to the target.
        for (index, cell) in self.indexed_iter() {
            let index = transforms.transform(&index);
            let ptr = target.index_mut(index);
            *ptr = *cell;
        }

        // since the empty cells are ignored, we are only adding more cells
        // thus we don't need to prune the chunks afterward
    }

    /// Copy a portion of the source gridmap to the target gridmap with the given transformation
    pub fn copy_to_within(
        &self,
        target: &mut Self,
        transforms: &[&dyn Transform<D>],
        bounding_box: &BoundingBox<D>,
    ) where
        A: Default + Copy,
        Ic: Eq + Hash + ConstZero + From<isize> + AsPrimitive<isize>,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
    {
        // For each cell in the bounded source gridmap,
        // transform the indexes and apply to the target.
        for (index, cell) in self.bounded_iter(*bounding_box) {
            let index = transforms.transform(&index);
            let ptr = target.index_mut(index);
            *ptr = *cell;
        }

        // since the empty cells are ignored, we are only adding more cells
        // thus we don't need to prune the chunks afterward
    }
}

impl<const D: usize> BoundingBox<D> {
    /// Check if the index is inside the specified boundaries
    pub fn contains(&self, index: &[isize; D]) -> bool {
        for (d, &i) in index.iter().enumerate() {
            if !(self.start[d] <= i && i < self.end[d]) {
                return false;
            }
        }
        true
    }

    /// Check if the two bounding boxes overlap
    pub fn overlaps_with(&self, other: &Self) -> bool {
        for d in 0..D {
            if !(self.start[d] <= other.end[d] && other.start[d] <= self.end[d]) {
                return false;
            }
        }
        true
    }
}
