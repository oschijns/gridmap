//! GridMap of cells

/// Indexing to access cells in the GridMap
pub mod indexing;

/// Iterator over the cells in the GridMap
pub mod iterator;

use crate::cell::Cell;
use hashbrown::HashMap;
use ndarray::{Array, Dim, Ix};

/// Chunk of cells
type Chunk<A, const D: usize> = Array<A, Dim<[Ix; D]>>;

/// GridMap of cells
pub struct GridMap<A, const D: usize, const S: Ix, Ic = isize>
where
    A: Cell,
{
    // TODO: check if the array should be boxed or not
    /// Internal data
    map: HashMap<[Ic; D], Chunk<A, D>>,

    /// Empty cell for out-of-bound access
    empty: A,
}

/// Create a new empty GridMap
impl<A, const D: usize, const S: Ix, Ic> Default for GridMap<A, D, S, Ic>
where
    A: Cell,
{
    #[inline]
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            empty: A::NULL,
        }
    }
}

impl<A, const D: usize, const S: Ix, Ic> GridMap<A, D, S, Ic>
where
    A: Cell,
{
    /// Create a new empty GridMap
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new GridMap with a predefined capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            empty: A::NULL,
        }
    }
}
