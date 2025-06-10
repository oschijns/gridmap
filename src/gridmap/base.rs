//! Basic operations available on the GridMap

use super::GridMap;
use crate::{cell::Cell, gridmap::make_chunk, util::is_chunk_empty};
use core::{hash::Hash, ops::Index};
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::{AsPrimitive, ConstZero, Euclid};

/// Access a cell in the gridmap
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    pub fn get<I>(&self, index: [I; D]) -> A
    where
        A: Clone,
        Ic: ConstZero + Eq + Hash + From<I>,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
        I: Euclid + From<Ix> + AsPrimitive<Ix>,
    {
        self.index(index).clone()
    }
}

/// Set a cell in the gridmap
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    pub fn set<I>(&mut self, index: [I; D], cell: A)
    where
        A: Clone + Default,
        Ic: ConstZero + Eq + Hash + From<I>,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
        I: Euclid + From<Ix> + AsPrimitive<Ix>,
    {
        // index of the chunk and index of the cell inside of the chunk
        let (chunk_index, cell_index) = self.split_index(index);

        if cell.is_null() {
            // remove a cell in the chunk
            // if the chunk does not exists, there is nothing to do
            if let Some(chunk) = self.map.get_mut(&chunk_index) {
                chunk[cell_index] = cell;

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
            chunk[cell_index] = cell;
        }
    }
}
