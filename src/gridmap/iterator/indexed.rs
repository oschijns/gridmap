//! Iterator over all non-empty cells with corresponding index

use crate::{
    cell::Cell,
    gridmap::{Chunk, GridMap},
};
use core::ops::AddAssign;
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::{AsPrimitive, ConstZero};

/// Get iterator over the grid map
impl<A, const D: usize, const S: Ix, Ic> GridMap<A, D, S, Ic>
where
    Ic: ConstZero,
    A: Cell,
{
    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn indexed_iter(&self) -> Iter<'_, A, D, S, Ic> {
        Iter {
            chunks: self.map.iter(),
            cells: None,
            cache: [0; D],
        }
    }

    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn indexed_iter_mut(&mut self) -> IterMut<'_, A, D, S, Ic> {
        IterMut {
            chunks: self.map.iter_mut(),
            cells: None,
            cache: [0; D],
        }
    }
}

/// Iterator over all the cells of the chunks of the GridMap
pub struct Iter<'i, A, const D: usize, const S: Ix, Ic = isize> {
    /// Iterator over the chunks
    chunks: hashbrown::hash_map::Iter<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::IndexedIter<'i, A, Dim<[Ix; D]>>>,

    /// Cache the index of the current chunk
    cache: [isize; D],
}

/// Access next element of the iterator
impl<'i, A, const D: usize, const S: Ix, Ic> Iterator for Iter<'i, A, D, S, Ic>
where
    A: Cell,
    Ic: Copy + AsPrimitive<isize>,
    Dim<[Ix; D]>: Dimension,
    [isize; D]: IntoDimension<Dim = Dim<[isize; D]>>,
    Dim<[isize; D]>: AddAssign<Dim<[Ix; D]>>,
{
    type Item = (Dim<[isize; D]>, &'i A);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for (cell_index, cell) in cells.by_ref() {
                    if !cell.is_null() {
                        let index = compute_index::<D, S>(&self.cache, cell_index);
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
pub struct IterMut<'i, A, const D: usize, const S: Ix, Ic = isize> {
    /// Iterator over the chunks
    chunks: hashbrown::hash_map::IterMut<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::IndexedIterMut<'i, A, Dim<[Ix; D]>>>,

    /// Cache the index of the current chunk
    cache: [isize; D],
}

/// Access next element of the iterator
impl<'i, A, const D: usize, const S: Ix, Ic> Iterator for IterMut<'i, A, D, S, Ic>
where
    A: Cell,
    Ic: Copy + AsPrimitive<isize>,
    Dim<[Ix; D]>: Dimension,
    [isize; D]: IntoDimension<Dim = Dim<[isize; D]>>,
    Dim<[isize; D]>: AddAssign<Dim<[Ix; D]>>,
{
    type Item = (Dim<[isize; D]>, &'i mut A);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for (cell_index, cell) in cells.by_ref() {
                    if !cell.is_null() {
                        let index = compute_index::<D, S>(&self.cache, cell_index);
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
fn compute_index<const D: usize, const S: usize>(
    chunk_index: &[isize; D],
    cell_index: <Dim<[Ix; D]> as Dimension>::Pattern,
) -> Dim<[isize; D]>
where
    Dim<[Ix; D]>: Dimension,
    [isize; D]: IntoDimension<Dim = Dim<[isize; D]>>,
    Dim<[isize; D]>: AddAssign<Dim<[Ix; D]>>,
{
    let cell_index: Dim<[Ix; D]> = cell_index.into_dimension();
    let mut index: Dim<[isize; D]> = Dim(chunk_index.map(|c| c * S as isize));
    index += cell_index;
    index
}
