//! Iterator over the cells in the GridMap

/// Iterator over all the cells of the chunks of the GridMap
pub mod simple;

/// Iterator over all non-empty cells of the GridMap
pub mod occupied;

/// Iterator over all non-empty cells with corresponding index
pub mod indexed;

/// Iterator over multiple neighboring cells
pub mod kernel;
