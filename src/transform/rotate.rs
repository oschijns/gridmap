//! Rotate indexes

use super::Transform;

/// Define the four possible orientations
#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Rotation {
    /// No rotation
    None = 0,

    /// Rotate by 90°
    Quarter = 1,

    /// Rotate by 180°
    Half = 2,

    /// Rotate by -90°
    ThreeQuarters = 3,
}

/// Define a rotation axis in 3D
#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Axis3D {
    X = 0,
    Y = 1,
    Z = 2,
}

/// Rotate the index
pub struct Rotate2(pub Rotation);

/// Rotate the index
pub struct Rotate3 {
    /// Rotation to apply
    pub rotation: Rotation,

    /// Axis to rotate around
    pub axis: Axis3D,

    /// Is the coordinate system left-handed ?
    pub left_handed: bool,
}

/// Apply the transformation to the given index
impl Transform<2> for Rotate2 {
    fn apply(&self, index: &mut [isize; 2]) {
        let [x, y] = *index;

        match self.0 {
            // 180°
            Rotation::Half => {
                index[0] = -x;
                index[1] = -y;
            }

            // Quarter rotation
            Rotation::Quarter => {
                index[0] = -y;
                index[1] = x;
            }
            Rotation::ThreeQuarters => {
                index[0] = y;
                index[1] = -x;
            }

            _ => {
                // if rotation is none, do nothing
            }
        }
    }
}

/// Apply the transformation to the given index
impl Transform<3> for Rotate3 {
    fn apply(&self, index: &mut [isize; 3]) {
        let [x, y, z] = *index;

        match (self.axis, self.rotation, self.left_handed) {
            // 180°
            (Axis3D::X, Rotation::Half, _) => {
                index[1] = -y;
                index[2] = -z;
            }
            (Axis3D::Y, Rotation::Half, _) => {
                index[0] = -x;
                index[2] = -z;
            }
            (Axis3D::Z, Rotation::Half, _) => {
                index[0] = -x;
                index[1] = -y;
            }

            // X-axis
            (Axis3D::X, Rotation::Quarter, false) | (Axis3D::X, Rotation::ThreeQuarters, true) => {
                index[1] = -z;
                index[2] = y;
            }
            (Axis3D::X, Rotation::Quarter, true) | (Axis3D::X, Rotation::ThreeQuarters, false) => {
                index[1] = z;
                index[2] = -y;
            }

            // Y-axis
            (Axis3D::Y, Rotation::Quarter, false) | (Axis3D::Y, Rotation::ThreeQuarters, true) => {
                index[0] = z;
                index[2] = -x;
            }
            (Axis3D::Y, Rotation::Quarter, true) | (Axis3D::Y, Rotation::ThreeQuarters, false) => {
                index[0] = -z;
                index[2] = x;
            }

            // Z-axis
            (Axis3D::Z, Rotation::Quarter, false) | (Axis3D::Z, Rotation::ThreeQuarters, true) => {
                index[0] = -y;
                index[1] = x;
            }
            (Axis3D::Z, Rotation::Quarter, true) | (Axis3D::Z, Rotation::ThreeQuarters, false) => {
                index[0] = y;
                index[1] = -x;
            }

            // No rotation
            _ => {
                // if rotation is none, do nothing
            }
        }
    }
}
