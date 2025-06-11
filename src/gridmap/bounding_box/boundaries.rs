//! Compute the boundaries of the gridmap

use super::BoundingBox;
use crate::{cell::Cell, gridmap::GridMap};
use ndarray::{Dim, Dimension, IntoDimension, Ix};
use num_traits::AsPrimitive;

impl<A, const D: usize, Ic> GridMap<A, D, Ic>
where
    A: Cell,
{
    /// Find the boundaries of the gridmap assuming empty chunks have been cleaned up.
    pub fn boundaries(&self) -> BoundingBox<D>
    where
        A: Clone,
        Ic: AsPrimitive<isize>,
        Dim<[Ix; D]>: Dimension,
    {
        // Prepare the two points to find.
        let mut chunk_0 = [isize::MAX; D];
        let mut chunk_1 = [isize::MIN; D];

        // Iterate over the chunks to find the extreme points in chunk coordinates.
        for (chunk_index, _) in self.map.iter() {
            // For each dimension, check if the chunk is further away.
            // If so, register it as a new extreme point.
            for d in 0..D {
                let c = chunk_index[d].as_();

                let p_0 = &mut chunk_0[d];
                *p_0 = c.min(*p_0);

                let p_1 = &mut chunk_1[d];
                *p_1 = c.max(*p_1);
            }
        }

        // Prepare the two points to find.
        let mut cell_0 = [isize::MAX; D];
        let mut cell_1 = [isize::MIN; D];

        // Reiterate over the chunks but this time look of the extreme cells.
        for (chunk_index, chunk) in self.map.iter() {
            // For each dimension, check if the chunk is an extreme one.
            // Check the cells in the chunk to find the extreme cell.
            for d in 0..D {
                let c = chunk_index[d].as_();

                // Check if the chunk is extreme towards -Inf
                let l_0 = chunk_0[d];
                if c == l_0 {
                    // Prepare a reference to edit the cell and get the position of the chunk.
                    let p = &mut cell_0[d];
                    let l = l_0 * self.chunk_dim[d] as isize;

                    // Iterate the cells to find a new extreme.
                    for (i, a) in chunk.indexed_iter() {
                        if !a.is_null() {
                            let i = i.into_dimension()[d] as isize + l;
                            *p = i.min(*p);
                        }
                    }
                }

                // Check if the chunk is extreme towards+Inf
                let l_1 = chunk_1[d];
                if c == l_1 {
                    // Prepare a reference to edit the cell and get the position of the chunk.
                    let p = &mut cell_1[d];
                    let l = l_1 * self.chunk_dim[d] as isize;

                    // Iterate the cells to find a new extreme.
                    for (i, a) in chunk.indexed_iter() {
                        if !a.is_null() {
                            let i = i.into_dimension()[d] as isize + l;
                            *p = i.max(*p);
                        }
                    }
                }
            }
        }

        BoundingBox {
            start: cell_0,
            end: cell_1,
        }
    }
}
