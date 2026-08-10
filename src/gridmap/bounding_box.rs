//! Bounding box module

use crate::{cell::Cell, gridmap::GridMap, transform::Transform};
use core::ops::IndexMut;
use ndarray::{Dim, Dimension, IntoDimension, Ix};

/// Compute the boundaries of the gridmap
pub mod boundaries;

/// Boundaries
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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

impl<A, const D: usize> GridMap<A, D>
where
    A: Cell,
{
    /// Copy a portion of the source gridmap to the target gridmap with the given transformation
    pub fn copy_to(&self, target: &mut Self, transforms: &[&dyn Transform<D>])
    where
        A: Default + Copy,
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
    /// Get the dimensions of the bounding box along each of its axis
    pub fn dimensions(&self) -> [usize; D] {
        let mut dim = [0; D];
        for ((&s, &e), item) in self.start.iter().zip(self.end.iter()).zip(dim.iter_mut()) {
            *item = (e - s) as usize;
        }
        dim
    }

    /// Check if the index is inside the specified boundaries
    pub fn contains(&self, index: &[isize; D]) -> bool {
        for ((&s, &e), &i) in self.start.iter().zip(self.end.iter()).zip(index) {
            if !(s <= i && i < e) {
                return false;
            }
        }
        true
    }

    /// Check if the two bounding boxes overlap
    pub fn overlaps_with(&self, other: &Self) -> bool {
        for d in 0..D {
            let ss = self.start[d];
            let se = self.end[d];
            let os = other.start[d];
            let oe = other.end[d];
            if !(ss <= oe && os <= se) {
                return false;
            }
        }
        true
    }

    /// Grow the bounding box with the provided index
    pub fn grow_with(&mut self, index: &[isize; D]) {
        for ((s, e), &i) in self.start.iter_mut().zip(self.end.iter_mut()).zip(index) {
            if *s > i {
                *s = i;
            }
            if *e < i {
                *e = i;
            }
        }
    }

    /// Grow the bounding box with the other bounding box
    pub fn grow_with_box(&mut self, other: &Self) {
        for (a, b) in self.start.iter_mut().zip(other.start) {
            if *a > b {
                *a = b;
            }
        }
        for (a, b) in self.end.iter_mut().zip(other.end) {
            if *a < b {
                *a = b;
            }
        }
    }
}
