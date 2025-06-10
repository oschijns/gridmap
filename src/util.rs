//! Utility functions

use crate::{Chunk, cell::Cell};
use ndarray::{Dim, Dimension, Ix};

/// Return true if the provided chunk contains only null cells
pub(crate) fn is_chunk_empty<A, const D: usize>(chunk: &Chunk<A, D>) -> bool
where
    A: Cell,
    Dim<[Ix; D]>: Dimension,
{
    for cell in chunk.iter() {
        if !cell.is_null() {
            return false;
        }
    }
    true
}
