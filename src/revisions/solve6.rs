#![allow(dead_code)]

use {
    added::*,
    com::*,
    comb::*,
    cropped::*,
    dim::*,
    n::*,
    ndarray::{Array3, Axis, Dimension, Ix3},
    padded::*,
    perm::*,
    std::collections::HashMap,
    transformed::*,
};

pub fn solve(n: usize) {
    let mut out = HashMap::new();

    let base = Cropped::default();

    recurse(&base, &mut out, n);

    let mut by_n = std::collections::BTreeMap::<N, Vec<Transformed>>::new();

    for (k, v) in out {
        if v {
            by_n.entry(N::from_array(&k)).or_default().push(k);
        }
    }

    for (k, v) in by_n {
        let v = v
            .into_iter()
            .map(Into::<Array3<bool>>::into)
            .collect::<Vec<_>>();
        println!("n: {:?}, count: {:?}", *k, v.len());
        //println!("{:?}", v);
    }

    fn recurse(cropped: &Cropped, out: &mut HashMap<Transformed, bool>, n: usize) {
        if *N::from_array(cropped) > n {
            return;
        }

        let mut transformations = Transformed::from_cropped(cropped);

        if let Some(canon) = transformations.next() {
            if let std::collections::hash_map::Entry::Vacant(entry) = out.entry(canon) {
                entry.insert(true);
            } else {
                return;
            }
        }

        for t in transformations {
            out.entry(t).or_insert(false);
        }

        let padded = Padded::from_cropped(cropped);

        for add_ix in Padded::empty_connected(&padded) {
            let added = Added::new(&padded, std::iter::once(add_ix));
            recurse(&Cropped::from_array(&added), out, n)
        }
    }
}

macro_rules! wrapper_for {
    ($Outer:ty, $Inner:ty) => {
        impl std::ops::Deref for $Outer {
            type Target = $Inner;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl From<$Outer> for $Inner {
            fn from(value: $Outer) -> Self {
                value.0
            }
        }
    };
}

mod cropped {
    use super::*;

    pub struct Cropped(Array3<bool>);

    impl Cropped {
        // FIXME: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            let (ixmin, ixmax) =
                get_bounding_box(array.indexed_iter().filter_map(|((x, y, z), &b)| {
                    if b {
                        Some(Ix3(x, y, z))
                    } else {
                        None
                    }
                }));

            let (xmin, ymin, zmin) = ixmin.into_pattern();
            let (xmax, ymax, zmax) = ixmax.into_pattern();

            let array = array
                .slice(ndarray::s![xmin..=xmax, ymin..=ymax, zmin..=zmax])
                .to_owned();

            Self(array)
        }

        pub fn transform(cropped: &Self, perm: &Permutation, comb: &Combination) -> Self {
            let mut view = cropped.view().permuted_axes(**perm);
            for (i, s) in comb.into_iter().enumerate() {
                if !s {
                    view.invert_axis(Axis(i))
                }
            }
            Self(view.to_owned())
        }
    }

    impl Default for Cropped {
        fn default() -> Self {
            Self(Array3::from_elem((1, 1, 1), true))
        }
    }

    wrapper_for!(Cropped, Array3<bool>);
}

mod padded {
    use super::*;

    pub struct Padded(Array3<bool>);

    pub struct AddIx(Ix3);

    impl Padded {
        pub fn from_cropped(cropped: &Cropped) -> Self {
            let mut array =
                Array3::from_elem(*Dimensions::from_array(cropped) + Ix3(2, 2, 2), false);
            #[allow(clippy::reversed_empty_ranges)]
            let mut slice = array.slice_mut(ndarray::s![1..-1, 1..-1, 1..-1]);
            slice.assign(cropped);
            Self(array)
        }

        pub fn empty_connected(padded: &Self) -> impl Iterator<Item = AddIx> + '_ {
            padded.indexed_iter().filter_map(|((x, y, z), &b)| {
                let ix = Ix3(x, y, z);
                if !b
                    && neighbor_ixs(ix)
                        .into_iter()
                        .any(|jx| padded.get(jx).copied().unwrap_or(false))
                {
                    Some(AddIx(ix))
                } else {
                    None
                }
            })
        }
    }

    wrapper_for!(Padded, Array3<bool>);
    wrapper_for!(AddIx, Ix3);
}

mod added {
    use super::*;

    pub struct Added(Array3<bool>);

    impl Added {
        pub fn new(padded: &Padded, add_ixs: impl Iterator<Item = AddIx>) -> Self {
            let mut array = (*padded).clone();
            for ix in add_ixs {
                array[*ix] = true;
            }
            Self(array)
        }
    }

    wrapper_for!(Added, Array3<bool>);
}

mod transformed {
    use super::*;

    #[derive(Hash, PartialEq, Eq)]
    pub struct Transformed(Array3<bool>);

    impl Transformed {
        pub fn from_cropped(cropped: &Cropped) -> impl Iterator<Item = Self> + '_ {
            Permutation::all().flat_map(move |p| {
                Combination::all().filter_map(move |c| {
                    let transformed = Cropped::transform(cropped, &p, &c);
                    let dim = Dimensions::from_array(&transformed);
                    let com = CenterOfMass::from_array(&transformed);
                    let n = N::from_array(&transformed);

                    if Dimensions::is_valid(&dim) && CenterOfMass::is_valid(&com, &dim, &n) {
                        Some(Self(transformed.clone()))
                    } else {
                        None
                    }
                })
            })
        }
    }

    wrapper_for!(Transformed, Array3<bool>);
}

mod dim {
    use super::*;

    pub struct Dimensions(Ix3);

    impl Dimensions {
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self(array.raw_dim())
        }

        pub fn is_valid(dim: &Self) -> bool {
            dim[0] >= dim[1] && dim[1] >= dim[2]
        }
    }

    wrapper_for!(Dimensions, Ix3);
}

mod com {
    use super::*;

    pub struct CenterOfMass(Ix3);

    impl CenterOfMass {
        // FIXME: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self(
                array
                    .indexed_iter()
                    .filter_map(|((x, y, z), &b)| b.then_some(Ix3(x, y, z)))
                    .reduce(|a, b| a + b)
                    .unwrap(),
            )
        }

        pub fn is_valid(com: &Self, dim: &Dimensions, n: &N) -> bool {
            let (dx, dy, dz) = (**dim - Ix3(1, 1, 1)).into_pattern();
            let (cx, cy, cz) = com.into_pattern();

            if 2 * cx > **n * dx || 2 * cy > **n * dy || 2 * cz > **n * dz {
                return false;
            }

            let xy = cy * dx <= dy * cx;
            let yz = cz * dy <= dz * cy;

            match (dx == dy, dy == dz) {
                (false, false) => true,
                (true, false) => xy,
                (false, true) => yz,
                (true, true) => xy && yz,
            }
        }
    }

    wrapper_for!(CenterOfMass, Ix3);
}

mod perm {
    pub struct Permutation([usize; 3]);

    impl Permutation {
        pub fn all() -> impl Iterator<Item = Self> {
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 0, 1],
                [0, 2, 1],
                [1, 2, 0],
                [2, 1, 0],
            ]
            .into_iter()
            .map(Self)
        }
    }

    wrapper_for!(Permutation, [usize; 3]);
}

mod comb {
    pub struct Combination([bool; 3]);

    impl Combination {
        pub fn all() -> impl Iterator<Item = Self> {
            [
                [false, false, false],
                [true, false, false],
                [false, true, false],
                [false, false, true],
                [true, true, false],
                [true, false, true],
                [false, true, true],
                [true, true, true],
            ]
            .into_iter()
            .map(Self)
        }
    }

    wrapper_for!(Combination, [bool; 3]);
}

mod n {
    use super::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    pub struct N(usize);

    impl N {
        // FIXME: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self(array.iter().filter(|&&b| b).count())
        }
    }

    wrapper_for!(N, usize);
}

fn get_bounding_box(ixs: impl Iterator<Item = Ix3>) -> (Ix3, Ix3) {
    use std::cmp::{max, min};

    let (ixmin, ixmax) = ixs.map(|ix| ix.into_pattern()).fold(
        (
            (usize::MAX, usize::MAX, usize::MAX),
            (usize::MIN, usize::MIN, usize::MIN),
        ),
        |(ixmin, ixmax), ix| {
            (
                (min(ixmin.0, ix.0), min(ixmin.1, ix.1), min(ixmin.2, ix.2)),
                (max(ixmax.0, ix.0), max(ixmax.1, ix.1), max(ixmax.2, ix.2)),
            )
        },
    );

    (
        Ix3(ixmin.0, ixmin.1, ixmin.2),
        Ix3(ixmax.0, ixmax.1, ixmax.2),
    )
}

fn neighbor_ixs(ix: Ix3) -> [Ix3; 6] {
    let (x, y, z) = ix.into_pattern();
    [
        Ix3(x.wrapping_add(1), y, z),
        Ix3(x, y.wrapping_add(1), z),
        Ix3(x, y, z.wrapping_add(1)),
        Ix3(x.wrapping_sub(1), y, z),
        Ix3(x, y.wrapping_sub(1), z),
        Ix3(x, y, z.wrapping_sub(1)),
    ]
}
