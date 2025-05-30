//! Iterator over all non-empty cells of the GridMap

use crate::{
    cell::Cell,
    gridmap::{Chunk, GridMap},
};
use ndarray::{Dim, Dimension, Ix};

/// Get iterator over the grid map
impl<A, const D: usize, const S: Ix, Ic> GridMap<A, D, S, Ic>
where
    A: Cell,
{
    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn iter(&self) -> Iter<'_, A, D, S, Ic> {
        Iter {
            chunks: self.map.iter(),
            cells: None,
        }
    }

    /// Create an iterator over all the cells of the chunks of the GridMap
    pub fn iter_mut(&mut self) -> IterMut<'_, A, D, S, Ic> {
        IterMut {
            chunks: self.map.iter_mut(),
            cells: None,
        }
    }
}

/// Iterator over all the cells of the chunks of the GridMap
pub struct Iter<'i, A, const D: usize, const S: Ix, Ic = isize> {
    /// Iterator over the chunks
    chunks: hashbrown::hash_map::Iter<'i, [Ic; D], Chunk<A, D>>,

    /// Iterator over the cells of the current chunk
    cells: Option<ndarray::iter::Iter<'i, A, Dim<[Ix; D]>>>,
}

/// Access next element of the iterator
impl<'i, A, const D: usize, const S: Ix, Ic> Iterator for Iter<'i, A, D, S, Ic>
where
    A: Cell,
    Dim<[Ix; D]>: Dimension,
{
    type Item = &'i A;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for cell in cells.by_ref() {
                    if !cell.is_null() {
                        return Some(cell);
                    }
                }
            }

            // Get an iterator over the next chunk
            if let Some((_, chunk)) = self.chunks.next() {
                self.cells = Some(chunk.iter());
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
    cells: Option<ndarray::iter::IterMut<'i, A, Dim<[Ix; D]>>>,
}

/// Access next element of the iterator
impl<'i, A, const D: usize, const S: Ix, Ic> Iterator for IterMut<'i, A, D, S, Ic>
where
    A: Cell,
    Dim<[Ix; D]>: Dimension,
{
    type Item = &'i mut A;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Do we have an iterator over the cells of the current chunk?
            if let Some(cells) = &mut self.cells {
                // Try to find a cell that is not null
                for cell in cells.by_ref() {
                    if !cell.is_null() {
                        return Some(cell);
                    }
                }
            }

            // Get an iterator over the next chunk
            if let Some((_, chunk)) = self.chunks.next() {
                self.cells = Some(chunk.iter_mut());
            } else {
                return None;
            }
        }
    }
}
