//! Indexing to access cells in the GridMap

use super::GridMap;
use crate::cell::Cell;
use core::{
    hash::Hash,
    ops::{Index, IndexMut},
};
use ndarray::{Array, Dim, Dimension, IntoDimension, Ix};
use num_traits::{AsPrimitive, Euclid};

/// Indexing to access cells in the GridMap
impl<A, const D: usize, const S: Ix, Ic, I> Index<[I; D]> for GridMap<A, D, S, Ic>
where
    A: Cell,
    Ic: Eq + Hash + From<I>,
    I: Euclid + From<Ix> + AsPrimitive<Ix>,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    type Output = A;

    /// Get a reference to the cell at the given index
    fn index(&self, index: [I; D]) -> &Self::Output {
        let (chunk_index, cell_index) = Self::split_index(index);
        self.index_chunk_cell(&chunk_index, cell_index)
    }
}

/// Indexing to mutable access cells in the GridMap
impl<A, const D: usize, const S: Ix, Ic, I> IndexMut<[I; D]> for GridMap<A, D, S, Ic>
where
    A: Cell + Default,
    Ic: Eq + Hash + From<I>,
    I: Euclid + From<Ix> + AsPrimitive<Ix>,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    /// Get a mutable reference to the cell at the given index
    fn index_mut(&mut self, index: [I; D]) -> &mut Self::Output {
        let (chunk_index, cell_index) = Self::split_index(index);
        self.index_chunk_cell_mut(chunk_index, cell_index)
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize, const S: Ix, Ic> GridMap<A, D, S, Ic>
where
    A: Cell,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
{
    /// Split the index into chunk index and cell index
    #[inline]
    pub fn split_index<I>(index: [I; D]) -> ([Ic; D], Dim<[Ix; D]>)
    where
        Ic: From<I>,
        I: Euclid + From<Ix> + AsPrimitive<Ix>,
    {
        let dim = I::from(S);
        let data = index.map(|i| i.div_rem_euclid(&dim));
        let chunk_index = data.map(|(q, _)| Ic::from(q));
        let cell_index = data.map(|(_, r)| r.as_());
        (chunk_index, Dim(cell_index))
    }
}

/// Index a cell knowing chunk index and cell index
impl<A, const D: usize, const S: usize, Ic> GridMap<A, D, S, Ic>
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
impl<A, const D: usize, const S: usize, Ic> GridMap<A, D, S, Ic>
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
            .or_insert_with(make_chunk::<A, D, S>);
        chunk.index_mut(cell_index)
    }
}

fn make_chunk<A, const D: usize, const S: Ix>() -> Array<A, Dim<[Ix; D]>>
where
    A: Default,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    Array::default(Dim([S; D]))
}
