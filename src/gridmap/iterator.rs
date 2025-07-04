//! Iterator over the cells in the GridMap

use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::AsPrimitive;

/// Iterator over all the cells of the chunks of the GridMap
pub mod simple;

/// Iterator over all non-empty cells of the GridMap
pub mod occupied;

/// Iterator over all non-empty cells with corresponding index
pub mod indexed;

/// Iterator over all non-empty cells within given boundaries with corresponding index
pub mod bounded;

/// Compute an index from a chunk index and a cell index.
#[inline]
fn from_chunk_to_cell_index<const D: usize, Ic>(
    chunk_dim: &[Ix; D],
    chunk_index: &[Ic; D],
) -> [isize; D]
where
    Ic: AsPrimitive<isize>,
{
    // prepare an index to construct
    let mut index = [0; D];
    for d in 0..D {
        index[d] = chunk_index[d].as_() * chunk_dim[d] as isize;
    }

    index
}

/// Compute an index from a chunk index and a cell index.
#[inline]
fn compute_cell_index<const D: usize>(
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
    for d in 0..D {
        index[d] = chunk_index[d] + cell_index[d] as isize;
    }

    index
}
