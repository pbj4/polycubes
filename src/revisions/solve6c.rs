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

#[allow(dead_code)]
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
            recurse(&Cropped::from_added(added), out, n)
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

    pub struct Cropped(Array3<bool>, N, CenterOfMass);

    impl Cropped {
        pub fn from_added(added: Added) -> Self {
            use std::cmp::{max, min};

            let n = Added::n(&added);

            let (xd, yd, zd) = Dimensions::from_array(&added).into_pattern();

            let (amin, amax) = Added::bounds(&added);
            let (xmin, ymin, zmin) = amin.into_pattern();
            let (xmax, ymax, zmax) = amax.into_pattern();

            let (xmin, ymin, zmin) = (min(xmin, 1), min(ymin, 1), min(zmin, 1));
            let (xmax, ymax, zmax) = (max(xmax, xd - 2), max(ymax, yd - 2), max(zmax, zd - 2));

            let com = CenterOfMass::to_cropped_from_added(&added);

            let array = Into::<Array3<bool>>::into(added).slice_move(ndarray::s![
                xmin..=xmax,
                ymin..=ymax,
                zmin..=zmax
            ]);

            debug_assert_eq!(com, CenterOfMass::from_array(&array));

            Self(array, n, com)
        }

        pub fn transform(cropped: &Self, perm: &Permutation, comb: &Combination) -> Self {
            let com = CenterOfMass::from_cropped_transform(cropped, perm, comb);

            let mut view = cropped.view().permuted_axes(**perm);
            for (i, s) in comb.into_iter().enumerate() {
                if !s {
                    view.invert_axis(Axis(i))
                }
            }
            let array = view.to_owned();

            debug_assert_eq!(com, CenterOfMass::from_array(&array));

            Self(array, cropped.1, com)
        }

        pub fn n(cropped: &Self) -> N {
            cropped.1
        }

        pub fn com(cropped: &Self) -> CenterOfMass {
            cropped.2
        }
    }

    impl Default for Cropped {
        fn default() -> Self {
            let array = Array3::from_elem((1, 1, 1), true);
            let n = N::from_array(&array);
            let com = CenterOfMass::from_array(&array);
            Self(array, n, com)
        }
    }

    wrapper_for!(Cropped, Array3<bool>);
}

mod padded {
    use super::*;

    pub struct Padded(Array3<bool>, N, CenterOfMass);

    #[derive(Clone)]
    pub struct AddIx(Ix3);

    impl Padded {
        pub fn from_cropped(cropped: &Cropped) -> Self {
            let com = CenterOfMass::to_padded_from_cropped(cropped);

            let mut array =
                Array3::from_elem(*Dimensions::from_array(cropped) + Ix3(2, 2, 2), false);
            #[allow(clippy::reversed_empty_ranges)]
            let mut slice = array.slice_mut(ndarray::s![1..-1, 1..-1, 1..-1]);
            slice.assign(cropped);

            debug_assert_eq!(com, CenterOfMass::from_array(&array));

            Self(array, Cropped::n(cropped), com)
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

        pub fn n(padded: &Self) -> N {
            padded.1
        }

        pub fn com(padded: &Self) -> CenterOfMass {
            padded.2
        }
    }

    wrapper_for!(Padded, Array3<bool>);
    wrapper_for!(AddIx, Ix3);
}

mod added {
    use super::*;

    pub struct Added(Array3<bool>, (Ix3, Ix3), N, CenterOfMass);

    impl Added {
        pub fn new(padded: &Padded, add_ixs: impl Iterator<Item = AddIx> + Clone) -> Self {
            let bounds = get_bounding_box(add_ixs.clone().map(|ix| *ix));
            let n = N::add_ixs(Padded::n(padded), add_ixs.clone());
            let com = CenterOfMass::from_padded_add(padded, add_ixs.clone());

            let mut array = (*padded).clone();
            for ix in add_ixs {
                array[*ix] = true;
            }

            debug_assert_eq!(com, CenterOfMass::from_array(&array));

            Self(array, bounds, n, com)
        }

        pub fn bounds(added: &Self) -> (Ix3, Ix3) {
            added.1
        }

        pub fn n(added: &Self) -> N {
            added.2
        }

        pub fn com(added: &Self) -> CenterOfMass {
            added.3
        }
    }

    wrapper_for!(Added, Array3<bool>);
}

mod transformed {
    use super::*;

    #[derive(Hash, PartialEq, Eq)]
    pub struct Transformed(Array3<bool>, N);

    impl Transformed {
        pub fn from_cropped(cropped: &Cropped) -> impl Iterator<Item = Self> + '_ {
            Permutation::all().flat_map(move |p| {
                Combination::all().filter_map(move |c| {
                    let dim = Dimensions::from_array(cropped);

                    if !Dimensions::is_valid(&Dimensions::transform(&dim, &p)) {
                        return None;
                    }

                    let n = Cropped::n(cropped);

                    let com = CenterOfMass::from_cropped_transform(cropped, &p, &c);

                    if !CenterOfMass::is_valid(&com, &dim, &n) {
                        return None;
                    }

                    let transformed = Cropped::transform(cropped, &p, &c);

                    debug_assert_eq!(com, CenterOfMass::from_array(&transformed));

                    Some(Self(transformed.clone(), n))
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

        pub fn transform(dim: &Self, perm: &Permutation) -> Self {
            let [x, y, z] = **perm;
            Self(Ix3(dim[x], dim[y], dim[z]))
        }
    }

    wrapper_for!(Dimensions, Ix3);
}

mod com {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
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

        pub fn from_cropped_transform(
            cropped: &Cropped,
            perm: &Permutation,
            comb: &Combination,
        ) -> Self {
            let [px, py, pz] = **perm;
            let com = Cropped::com(cropped);
            let mut com = [com[px], com[py], com[pz]];

            let d: [usize; 3] = Dimensions::transform(&Dimensions::from_array(cropped), perm)
                .into_pattern()
                .into();

            let n = Cropped::n(cropped);

            for i in 0..3 {
                if !comb[i] {
                    com[i] = (d[i] - 1) * *n - com[i];
                }
            }

            let [cx, cy, cz] = com;

            Self(Ix3(cx, cy, cz))
        }

        pub fn to_cropped_from_added(added: &Added) -> Self {
            let (min, _) = Added::bounds(added);
            let (x, y, z) = min.into_pattern();
            let min = Ix3(x.min(1), y.min(1), z.min(1));

            Self(*Added::com(added) - min * *Added::n(added))
        }

        pub fn from_padded_add(padded: &Padded, add_ixs: impl Iterator<Item = AddIx>) -> Self {
            Self(*Padded::com(padded) + add_ixs.map(|a| *a).reduce(|a, b| a + b).unwrap())
        }

        pub fn to_padded_from_cropped(cropped: &Cropped) -> Self {
            Self(*Cropped::com(cropped) + Ix3(1, 1, 1) * *Cropped::n(cropped))
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

    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
    pub struct N(usize);

    impl N {
        // FIXME: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self(array.iter().filter(|&&b| b).count())
        }

        pub fn add_ixs(n: Self, add_ixs: impl Iterator<Item = AddIx>) -> Self {
            Self(*n + add_ixs.count())
        }
    }

    wrapper_for!(N, usize);
}

// (min, max)
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
