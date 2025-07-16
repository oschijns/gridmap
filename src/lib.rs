#![cfg_attr(not(feature = "std"), no_std)]

/// Use alloc crate for no_std support
extern crate alloc;
use hashbrown::HashMap;
use ndarray::{Array, Dim, Ix};

/// GridMap of cells
pub mod gridmap;

/// Trait to implement to cells inserted in the gridmap
pub mod cell;

/// Define how to transform a map
pub mod transform;

/// Utility functions
pub mod util;

/// GridMap of cells
pub struct GridMap<A, const D: usize, Ic = isize> {
    /// Dimensions of the chunks in the gridmap
    chunk_dim: [Ix; D],

    // TODO: check if the array should be boxed or not
    /// Internal data
    map: HashMap<[Ic; D], Chunk<A, D>>,

    /// Empty cell for out-of-bound access
    empty: A,
}

/// Chunk of cells
pub type Chunk<A, const D: usize> = Array<A, Dim<[Ix; D]>>;
