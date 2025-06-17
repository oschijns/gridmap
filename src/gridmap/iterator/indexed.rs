//! Iterator over all non-empty cells with corresponding index

use super::{WithNulls, compute_cell_index, from_chunk_to_cell_index};
use crate::{
    cell::Cell,
    gridmap::{Chunk, GridMap},
};
use ndarray::{Dim, Dimension, Ix};
use num_traits::{AsPrimitive, ConstZero};

/// Get iterator over the grid map
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    Ic: ConstZero,
    A: Cell,
{
    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn indexed_iter(&self) -> Iter<'_, A, D, Ic> {
        Iter {
            chunks: self.map.iter(),
            cells: None,
            skip_null: false,
            chunk_dim: self.chunk_dim,
            cache: [0; D],
        }
    }

    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn indexed_iter_mut(&mut self) -> IterMut<'_, A, D, Ic> {
        IterMut {
            chunks: self.map.iter_mut(),
            cells: None,
            skip_null: false,
            chunk_dim: self.chunk_dim,
            cache: [0; D],
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

    /// Cache the index of the current chunk in cell coordinates
    cache: [isize; D],
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

    /// Cache the index of the current chunk in cell coordinates
    cache: [isize; D],
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
        loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for (cell_index, cell) in cells.by_ref() {
                    if !self.skip_null || !cell.is_null() {
                        let index = compute_cell_index::<D>(&self.cache, cell_index);
                        return Some((index, cell));
                    }
                }
            }

            // Get an iterator over the next chunk
            if let Some((chunk_index, chunk)) = self.chunks.next() {
                self.cache = from_chunk_to_cell_index(&self.chunk_dim, chunk_index);
                self.cells = Some(chunk.indexed_iter());
            } else {
                return None;
            }
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
        loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for (cell_index, cell) in cells.by_ref() {
                    if !self.skip_null || !cell.is_null() {
                        let index = compute_cell_index::<D>(&self.cache, cell_index);
                        return Some((index, cell));
                    }
                }
            }

            // Get an iterator over the next chunk
            if let Some((chunk_index, chunk)) = self.chunks.next() {
                self.cache = from_chunk_to_cell_index(&self.chunk_dim, chunk_index);
                self.cells = Some(chunk.indexed_iter_mut());
            } else {
                return None;
            }
        }
    }
}
