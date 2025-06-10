#![cfg_attr(not(feature = "std"), no_std)]

/// Use alloc crate for no_std support
extern crate alloc;
use ndarray::{Array, Dim, Ix};

/// GridMap of cells
pub mod gridmap;

/// Trait to implement to cells inserted in the gridmap
pub mod cell;

/// Utility functions
pub mod util;

/// Chunk of cells
pub type Chunk<A, const D: usize> = Array<A, Dim<[Ix; D]>>;
