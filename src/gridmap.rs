//! GridMap of cells

/// Basic operations available on the GridMap
pub mod base;

/// Indexing to access cells in the GridMap
pub mod indexing;

/// Iterator over the cells in the GridMap
pub mod iterator;

/// Bounding box
pub mod bounding_box;

use crate::{Chunk, cell::Cell};
use hashbrown::HashMap;
use ndarray::{Array, Dim, Dimension, IntoDimension, Ix};

/// GridMap of cells
pub struct GridMap<A, const D: usize, Ic = isize>
where
    A: Cell,
{
    /// Dimensions of the chunks in the gridmap
    chunk_dim: [Ix; D],

    // TODO: check if the array should be boxed or not
    /// Internal data
    map: HashMap<[Ic; D], Chunk<A, D>>,

    /// Empty cell for out-of-bound access
    empty: A,
}

/// Create a new empty GridMap
impl<A, const D: usize, Ic> Default for GridMap<A, D, Ic>
where
    A: Cell,
{
    #[inline]
    fn default() -> Self {
        Self {
            chunk_dim: [12; D],
            map: HashMap::new(),
            empty: A::NULL,
        }
    }
}

impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    /// Create a new empty GridMap
    #[inline]
    pub fn new(chunk_dim: [Ix; D]) -> Self {
        Self {
            chunk_dim,
            map: HashMap::new(),
            empty: A::NULL,
        }
    }

    /// Create a new GridMap with a predefined capacity
    pub fn with_capacity(chunk_dim: [Ix; D], capacity: usize) -> Self {
        Self {
            chunk_dim,
            map: HashMap::with_capacity(capacity),
            empty: A::NULL,
        }
    }
}

/// Build a chunk with the given dimensions
fn make_chunk<A, const D: usize>(chunk_dim: &[Ix; D]) -> Array<A, Dim<[Ix; D]>>
where
    A: Default,
    [Ix; D]: IntoDimension<Dim = Dim<[Ix; D]>>,
    Dim<[Ix; D]>: Dimension,
{
    Array::default(Dim(*chunk_dim))
}
