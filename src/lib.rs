#![cfg_attr(not(feature = "std"), no_std)]

/// GridMap of cells
pub mod gridmap;

/// Trait to implement to cells inserted in the gridmap
pub mod cell;

/// Usealloc crate for no_std support
extern crate alloc;
