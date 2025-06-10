//! Indexing to access cells in the GridMap

use super::GridMap;
use crate::{cell::Cell, gridmap::make_chunk};
use core::{
    hash::Hash,
    ops::{Index, IndexMut},
};
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::{AsPrimitive, ConstZero, Euclid};

/// Indexing to access cells in the GridMap
impl<A, const D: usize, Ic, I> Index<[I; D]> for GridMap<A, D, Ic>
where
    A: Cell,
    Ic: ConstZero + Eq + Hash + From<I>,
    I: Euclid + From<Ix> + AsPrimitive<Ix>,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    type Output = A;

    /// Get a reference to the cell at the given index
    fn index(&self, index: [I; D]) -> &Self::Output {
        let (chunk_index, cell_index) = self.split_index(index);
        self.index_chunk_cell(&chunk_index, cell_index)
    }
}

/// Indexing to mutable access cells in the GridMap
impl<A, const D: usize, Ic, I> IndexMut<[I; D]> for GridMap<A, D, Ic>
where
    A: Cell + Default,
    Ic: ConstZero + Eq + Hash + From<I>,
    I: Euclid + From<Ix> + AsPrimitive<Ix>,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    /// Get a mutable reference to the cell at the given index
    fn index_mut(&mut self, index: [I; D]) -> &mut Self::Output {
        let (chunk_index, cell_index) = self.split_index(index);
        self.index_chunk_cell_mut(chunk_index, cell_index)
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Ic: ConstZero,
{
    /// Split the index into chunk index and cell index
    #[inline]
    pub fn split_index<I>(&self, index: [I; D]) -> ([Ic; D], Dim<[Ix; D]>)
    where
        Ic: From<I>,
        I: Euclid + From<Ix> + AsPrimitive<Ix>,
    {
        // prepare arrays to store the results
        let mut chunk_index = [Ic::ZERO; D];
        let mut cell_index = [Ix::ZERO; D];

        // for each component
        for i in 0..D {
            let (ch, cl) = index[i].div_rem_euclid(&self.chunk_dim[i].into());
            chunk_index[i] = Ic::from(ch);
            cell_index[i] = cl.as_();
        }
        (chunk_index, Dim(cell_index))
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
    Ic: Eq + Hash,
    Dim<[Ix; D]>: Dimension,
{
    /// Index a cell knowing chunk index and cell index
    pub fn index_chunk_cell<'m>(
        &'m self,
        chunk_index: &[Ic; D],
        cell_index: Dim<[Ix; D]>,
    ) -> &'m A {
        if let Some(chunk) = self.map.get(chunk_index) {
            chunk.index(cell_index)
        } else {
            &self.empty
        }
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell + Default,
    Ic: Eq + Hash,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    /// Index a cell knowing chunk index and cell index
    pub fn index_chunk_cell_mut(
        &mut self,
        chunk_index: [Ic; D],
        cell_index: Dim<[Ix; D]>,
    ) -> &mut A {
        let chunk = self
            .map
            .entry(chunk_index)
            .or_insert_with(|| make_chunk::<A, D>(&self.chunk_dim));
        chunk.index_mut(cell_index)
    }
}
