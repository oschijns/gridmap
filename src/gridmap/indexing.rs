//! Indexing to access cells in the GridMap

use crate::{GridMap, Index, cell::Cell, gridmap::make_chunk};
use core::ops;
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::{ConstZero, Euclid};

/// Indexing to access cells in the GridMap
impl<A, const D: usize> ops::Index<[Index; D]> for GridMap<A, D>
where
    A: Cell,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    type Output = A;

    /// Get a reference to the cell at the given index
    fn index(&self, index: [Index; D]) -> &Self::Output {
        let (chunk_index, cell_index) = self.split_index(&index);
        self.index_chunk_cell(&chunk_index, &cell_index)
    }
}

/// Indexing to mutable access cells in the GridMap
impl<A, const D: usize> ops::IndexMut<[Index; D]> for GridMap<A, D>
where
    A: Cell + Default,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    /// Get a mutable reference to the cell at the given index
    fn index_mut(&mut self, index: [Index; D]) -> &mut Self::Output {
        let (chunk_index, cell_index) = self.split_index(&index);
        self.index_chunk_cell_mut(chunk_index, &cell_index)
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize> GridMap<A, D>
where
    A: Cell,
{
    /// Index a cell knowing chunk index and cell index
    pub fn index_chunk_cell<'m>(
        &'m self,
        chunk_index: &[Index; D],
        cell_index: &Dim<[Ix; D]>,
    ) -> &'m A
    where
        Dim<[Ix; D]>: Dimension,
    {
        if let Some(chunk) = self.map.get(chunk_index) {
            ops::Index::index(chunk, *cell_index)
        } else {
            &self.empty
        }
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize> GridMap<A, D>
where
    A: Cell,
{
    /// Index a cell knowing chunk index and cell index
    pub fn index_chunk_cell_mut(
        &mut self,
        chunk_index: [Index; D],
        cell_index: &Dim<[Ix; D]>,
    ) -> &mut A
    where
        A: Default,
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
        Dim<[Ix; D]>: Dimension,
    {
        let chunk = self
            .map
            .entry(chunk_index)
            .or_insert_with(|| make_chunk::<A, D>(&self.chunk_dim));
        ops::IndexMut::index_mut(chunk, *cell_index)
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize> GridMap<A, D>
where
    A: Cell,
{
    /// Split the index into chunk index and cell index
    #[inline]
    pub fn split_index(&self, index: &[Index; D]) -> ([Index; D], Dim<[Ix; D]>)
    where
        [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    {
        // prepare arrays to store the results
        let mut chunk_index = [Index::ZERO; D];
        let mut cell_index = [Ix::ZERO; D];

        // for each component
        for i in 0..D {
            let idx = index[i];
            let dim = self.chunk_dim[i] as isize;

            let (ch, cl) = idx.div_rem_euclid(&dim);
            chunk_index[i] = ch;
            cell_index[i] = cl as Ix;
        }
        (chunk_index, Dim(cell_index))
    }
}
