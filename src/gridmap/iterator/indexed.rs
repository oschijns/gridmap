//! Iterator over all non-empty cells with corresponding index

use crate::{
    cell::Cell,
    gridmap::{Chunk, GridMap},
};
use ndarray::{Dim, Dimension, IntoDimension, Ix};
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
            chunk_dim: self.chunk_dim,
            chunks: self.map.iter(),
            cells: None,
            cache: [0; D],
        }
    }

    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn indexed_iter_mut(&mut self) -> IterMut<'_, A, D, Ic> {
        IterMut {
            chunk_dim: self.chunk_dim,
            chunks: self.map.iter_mut(),
            cells: None,
            cache: [0; D],
        }
    }
}

/// Iterator over all the cells of the chunks of the GridMap
pub struct Iter<'i, A, const D: usize, Ic = isize> {
    /// Dimensions of the chunks in the gridmap
    chunk_dim: [Ix; D],

    /// Iterator over the chunks
    chunks: hashbrown::hash_map::Iter<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::IndexedIter<'i, A, Dim<[Ix; D]>>>,

    /// Cache the index of the current chunk
    cache: [isize; D],
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
                    if !cell.is_null() {
                        let index = compute_index::<D>(&self.chunk_dim, &self.cache, cell_index);
                        return Some((index, cell));
                    }
                }
            }

            // Get an iterator over the next chunk
            if let Some((index, chunk)) = self.chunks.next() {
                self.cache = index.map(|c| c.as_());
                self.cells = Some(chunk.indexed_iter());
            } else {
                return None;
            }
        }
    }
}

/// Mutable Iiterator over all the cells of the chunks of the GridMap
pub struct IterMut<'i, A, const D: usize, Ic = isize> {
    /// Dimensions of the chunks in the gridmap
    chunk_dim: [Ix; D],

    /// Iterator over the chunks
    chunks: hashbrown::hash_map::IterMut<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::IndexedIterMut<'i, A, Dim<[Ix; D]>>>,

    /// Cache the index of the current chunk
    cache: [isize; D],
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
                    if !cell.is_null() {
                        let index = compute_index::<D>(&self.chunk_dim, &self.cache, cell_index);
                        return Some((index, cell));
                    }
                }
            }

            // Get an iterator over the next chunk
            if let Some((index, chunk)) = self.chunks.next() {
                self.cache = index.map(|c| c.as_());
                self.cells = Some(chunk.indexed_iter_mut());
            } else {
                return None;
            }
        }
    }
}

/// Compute an index from a chunk index and a cell index.
#[inline]
fn compute_index<const D: usize>(
    chunk_dim: &[Ix; D],
    chunk_index: &[isize; D],
    cell_index: <Dim<[Ix; D]> as Dimension>::Pattern,
) -> [isize; D]
where
    Dim<[Ix; D]>: Dimension,
{
    // convert cell index into an indexable form
    let cell_index: Dim<[Ix; D]> = cell_index.into_dimension();

    // prepare an index to construct
    let mut index = [0; D];
    for i in 0..D {
        index[i] = chunk_index[i] * chunk_dim[i] as isize + cell_index[i] as isize;
    }

    index
}
