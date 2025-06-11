//! Basic operations available on the GridMap

use super::GridMap;
use crate::{Chunk, cell::Cell, gridmap::make_chunk, util::is_chunk_empty};
use core::{hash::Hash, ops::IndexMut};
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::{AsPrimitive, ConstZero};

/// Access a cell in the gridmap
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    pub fn get<I>(&self, index: &[I; D]) -> A
    where
        A: Clone,
        Ic: Eq + Hash + ConstZero + From<isize>,
        I: AsPrimitive<isize>,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
    {
        let (chunk_index, cell_index) = self.split_index(index);
        self.index_chunk_cell(&chunk_index, &cell_index).clone()
    }
}

/// Set a cell in the gridmap
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    pub fn set<I>(&mut self, index: &[I; D], cell: A)
    where
        A: Default,
        Ic: Eq + Hash + ConstZero + From<isize>,
        I: AsPrimitive<isize>,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
    {
        // index of the chunk and index of the cell inside of the chunk
        let (chunk_index, cell_index) = self.split_index(index);

        if cell.is_null() {
            // remove a cell in the chunk
            // if the chunk does not exists, there is nothing to do
            if let Some(chunk) = self.map.get_mut(&chunk_index) {
                let ptr = chunk.index_mut(cell_index);
                *ptr = cell;

                // if the chunk end up empty, remove it from the map
                if is_chunk_empty(chunk) {
                    self.map.remove(&chunk_index);
                }
            }
        } else {
            // add a new cell in the chunk
            // if the chunk does not exists, create it
            let chunk = self
                .map
                .entry(chunk_index)
                .or_insert_with(|| make_chunk::<A, D>(&self.chunk_dim));

            // set the cell
            let ptr = chunk.index_mut(cell_index);
            *ptr = cell;
        }
    }
}

impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    /// Access a chunk
    #[inline]
    pub fn get_chunk<I>(&self, chunk_index: &[Ic; D]) -> Option<&Chunk<A, D>>
    where
        Ic: Eq + Hash,
        Dim<[Ix; D]>: Dimension,
    {
        self.map.get(chunk_index)
    }

    /// Access a chunk as mutable
    #[inline]
    pub fn get_chunk_mut<I>(&mut self, chunk_index: &[Ic; D]) -> Option<&mut Chunk<A, D>>
    where
        Ic: Eq + Hash,
        Dim<[Ix; D]>: Dimension,
    {
        self.map.get_mut(chunk_index)
    }

    /// Check if the chunk at given chunk index should be freed
    pub fn try_free_chunk<I>(&mut self, chunk_index: &[Ic; D]) -> bool
    where
        Ic: Eq + Hash,
        Dim<[Ix; D]>: Dimension,
    {
        // if the chunk does not exists, there is nothing to do
        if let Some(chunk) = self.map.get(chunk_index) {
            // if the chunk end up empty, remove it from the map
            if is_chunk_empty(chunk) {
                self.map.remove(chunk_index);
                return true;
            }
        }
        false
    }

    /// Clean up chunks that should be freed
    #[inline]
    pub fn prune(&mut self)
    where
        Ic: Eq + Hash,
        Dim<[Ix; D]>: Dimension,
    {
        self.map.retain(|_, chunk| !is_chunk_empty(chunk));
    }
}
