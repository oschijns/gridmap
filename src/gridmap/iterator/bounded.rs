//! Iterator over all non-empty cells with corresponding index

use super::{WithNulls, compute_cell_index, from_chunk_to_cell_index};
use crate::{
    cell::Cell,
    gridmap::{Chunk, GridMap, bounding_box::BoundingBox},
};
use ndarray::{Dim, Dimension, Ix};
use num_traits::{AsPrimitive, ConstZero};

/// Get iterator over the grid map
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
    Ic: ConstZero,
{
    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn bounded_iter(&self, bounds: BoundingBox<D>) -> Iter<'_, A, D, Ic> {
        Iter {
            chunks: self.map.iter(),
            cells: None,
            skip_null: false,
            chunk_dim: self.chunk_dim,
            cache: [0; D],
            bounds,
        }
    }

    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn bounded_iter_mut(&mut self, bounds: BoundingBox<D>) -> IterMut<'_, A, D, Ic> {
        IterMut {
            chunks: self.map.iter_mut(),
            cells: None,
            skip_null: false,
            chunk_dim: self.chunk_dim,
            cache: [0; D],
            bounds,
        }
    }
}

/// Iterator over all the cells of the chunks of the GridMap
pub struct Iter<'i, A, const D: usize, Ic = isize> {
    /// Iterator over the chunks
    chunks: hashbrown::hash_map::Iter<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::IndexedIter<'i, A, Dim<[Ix; D]>>>,

    /// Specify if null cells should be skipped
    skip_null: bool,

    /// Dimensions of the chunks in the gridmap
    chunk_dim: [Ix; D],

    /// Cache the index of the current chunk
    cache: [isize; D],

    /// Boundaries to look for cells
    bounds: BoundingBox<D>,
}

/// Mutable Iiterator over all the cells of the chunks of the GridMap
pub struct IterMut<'i, A, const D: usize, Ic = isize> {
    /// Iterator over the chunks
    chunks: hashbrown::hash_map::IterMut<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::IndexedIterMut<'i, A, Dim<[Ix; D]>>>,

    /// Specify if null cells should be skipped
    skip_null: bool,

    /// Dimensions of the chunks in the gridmap
    chunk_dim: [Ix; D],

    /// Cache the index of the current chunk
    cache: [isize; D],

    /// Boundaries to look for cells
    bounds: BoundingBox<D>,
}

/// Modify the iterator to include null cells in the iteration
impl<'i, A, const D: usize, Ic> WithNulls for Iter<'i, A, D, Ic> {
    fn with_nulls(&mut self) -> &Self {
        self.skip_null = true;
        self
    }
}

/// Modify the iterator to include null cells in the iteration
impl<'i, A, const D: usize, Ic> WithNulls for IterMut<'i, A, D, Ic> {
    fn with_nulls(&mut self) -> &Self {
        self.skip_null = true;
        self
    }
}

/// Access next element of the iterator
impl<'i, A, const D: usize, Ic> Iterator for Iter<'i, A, D, Ic>
where
    A: Cell,
    Ic: AsPrimitive<isize>,
    Dim<[Ix; D]>: Dimension,
{
    type Item = ([isize; D], &'i A);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for (cell_index, cell) in cells.by_ref() {
                    if !self.skip_null || !cell.is_null() {
                        let index = compute_cell_index::<D>(&self.cache, cell_index);
                        // TODO: create a view over the array to remove this check
                        if self.bounds.contains(&index) {
                            return Some((index, cell));
                        }
                    }
                }
            }

            // Get an iterator over the next chunk
            for (chunk_index, chunk) in &mut self.chunks {
                let index = from_chunk_to_cell_index(&self.chunk_dim, chunk_index);
                let bounds = chunk_bounds(&self.chunk_dim, &index);
                if self.bounds.overlaps_with(&bounds) {
                    self.cache = index;
                    // TODO: create a view over the array
                    self.cells = Some(chunk.indexed_iter());
                    continue 'outer;
                }
            }

            return None;
        }
    }
}

/// Access next element of the iterator
impl<'i, A, const D: usize, Ic> Iterator for IterMut<'i, A, D, Ic>
where
    A: Cell,
    Ic: AsPrimitive<isize>,
    Dim<[Ix; D]>: Dimension,
{
    type Item = ([isize; D], &'i mut A);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for (cell_index, cell) in cells.by_ref() {
                    if !self.skip_null || !cell.is_null() {
                        let index = compute_cell_index::<D>(&self.cache, cell_index);
                        // TODO: create a view over the array to remove this check
                        if self.bounds.contains(&index) {
                            return Some((index, cell));
                        }
                    }
                }
            }

            // Get an iterator over the next chunk
            for (chunk_index, chunk) in &mut self.chunks {
                let index = from_chunk_to_cell_index(&self.chunk_dim, chunk_index);
                let bounds = chunk_bounds(&self.chunk_dim, &index);
                if self.bounds.overlaps_with(&bounds) {
                    self.cache = index;
                    // TODO: create a view over the array
                    self.cells = Some(chunk.indexed_iter_mut());
                    continue 'outer;
                }
            }

            return None;
        }
    }
}

/// Compute the bounding box of the chunk
#[inline]
fn chunk_bounds<const D: usize>(chunk_dim: &[Ix; D], chunk_index: &[isize; D]) -> BoundingBox<D>
where
    Dim<[Ix; D]>: Dimension,
{
    // prepare the two points
    let mut start = [0; D];
    let mut end = [0; D];

    // for each dimension
    for i in 0..D {
        let d = chunk_dim[i] as isize;
        let s = chunk_index[i] * d;
        start[i] = s;
        end[i] = s + d;
    }

    BoundingBox { start, end }
}
