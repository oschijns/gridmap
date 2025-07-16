use crate::transform::{
    Transform,
    mirror::Mirror,
    rotate::{Rotate2, Rotate3},
    translate::Translate,
};

/// Combine transforms in 2D
#[derive(Debug, Default)]
pub struct Combined2 {
    /// First mirror the object
    pub mirror: Mirror<2>,

    /// Second rotate the object
    pub rotate: Rotate2,

    /// Third translate the object
    pub translate: Translate<2>,
}

/// Combine transforms in 3D
#[derive(Debug, Default)]
pub struct Combined3 {
    /// First mirror the object
    pub mirror: Mirror<3>,

    /// Second rotate the object
    pub rotate: Rotate3,

    /// Third translate the object
    pub translate: Translate<3>,
}

impl Transform<2> for Combined2 {
    fn apply(&self, index: &mut [isize; 2]) {
        self.mirror.apply(index);
        self.rotate.apply(index);
        self.translate.apply(index);
    }
}

impl Transform<3> for Combined3 {
    fn apply(&self, index: &mut [isize; 3]) {
        self.mirror.apply(index);
        self.rotate.apply(index);
        self.translate.apply(index);
    }
}

impl Combined2 {
    pub fn new(mirror: Mirror<2>, rotate: Rotate2, translate: Translate<2>) -> Self {
        Self {
            mirror,
            rotate,
            translate,
        }
    }

    // TODO: fuse two combined transform into one
}

impl Combined3 {
    pub fn new(mirror: Mirror<3>, rotate: Rotate3, translate: Translate<3>) -> Self {
        Self {
            mirror,
            rotate,
            translate,
        }
    }

    // TODO: fuse two combined transform into one
}

impl From<Mirror<2>> for Combined2 {
    fn from(mirror: Mirror<2>) -> Self {
        Self {
            mirror,
            ..Default::default()
        }
    }
}

impl From<Rotate2> for Combined2 {
    fn from(rotate: Rotate2) -> Self {
        Self {
            rotate,
            ..Default::default()
        }
    }
}

impl From<Translate<2>> for Combined2 {
    fn from(translate: Translate<2>) -> Self {
        Self {
            translate,
            ..Default::default()
        }
    }
}

impl From<Mirror<3>> for Combined3 {
    fn from(mirror: Mirror<3>) -> Self {
        Self {
            mirror,
            ..Default::default()
        }
    }
}

impl From<Rotate3> for Combined3 {
    fn from(rotate: Rotate3) -> Self {
        Self {
            rotate,
            ..Default::default()
        }
    }
}

impl From<Translate<3>> for Combined3 {
    fn from(translate: Translate<3>) -> Self {
        Self {
            translate,
            ..Default::default()
        }
    }
}
