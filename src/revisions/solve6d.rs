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
    std::collections::{BTreeMap, HashMap},
    transformed::*,
};

#[allow(dead_code)]
pub fn solve(n: usize) -> BTreeMap<usize, usize> {
    let mut out = HashMap::new();

    let base = Cropped::default();

    recurse(&base, &mut out, n);

    let mut by_n = BTreeMap::<N, Vec<Transformed>>::new();

    for (k, v) in out {
        if v {
            by_n.entry(N::from_array(k.array())).or_default().push(k);
        }
    }

    let counts = by_n
        .iter()
        .map(|(k, v)| (k.num(), v.len()))
        .collect::<BTreeMap<_, _>>();

    for (k, v) in by_n {
        let v = v.into_iter().map(|t| t.array().clone()).collect::<Vec<_>>();
        println!("n: {:?}, count: {:?}", k.num(), v.len());
        //println!("{:?}", v);
    }

    counts
}

fn recurse(cropped: &Cropped, out: &mut HashMap<Transformed, bool>, n: usize) {
    if N::from_array(cropped.array()).num() > n {
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

    for add_ix in padded.empty_connected() {
        let added = Added::from_padded(&padded, std::iter::once(add_ix));
        recurse(&Cropped::from_array(added.array()), out, n)
    }
}

trait HasArray {
    fn array(&self) -> &Array3<bool>;

    fn dim(&self) -> Dimensions {
        Dimensions::from_array(self.array())
    }
}

trait HasIx {
    fn ix(&self) -> Ix3;
}

mod cropped {
    use super::*;

    pub struct Cropped {
        array: Array3<bool>,
    }

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

            Self { array }
        }

        pub fn transform(&self, perm: &Permutation, comb: &Combination) -> Self {
            let mut view = self.array.view().permuted_axes(perm.indicies());
            for (i, s) in comb.mask().into_iter().enumerate() {
                if !s {
                    view.invert_axis(Axis(i))
                }
            }
            let array = view.to_owned();

            Self { array }
        }
    }

    impl Default for Cropped {
        fn default() -> Self {
            Self {
                array: Array3::from_elem((1, 1, 1), true),
            }
        }
    }

    impl HasArray for Cropped {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }
    }
}

mod padded {
    use super::*;

    pub struct Padded {
        array: Array3<bool>,
    }

    pub struct AddIx {
        ix: Ix3,
    }

    impl Padded {
        pub fn from_cropped(cropped: &Cropped) -> Self {
            let mut array = Array3::from_elem(cropped.dim().ix() + Ix3(2, 2, 2), false);
            #[allow(clippy::reversed_empty_ranges)]
            let mut slice = array.slice_mut(ndarray::s![1..-1, 1..-1, 1..-1]);
            slice.assign(cropped.array());
            Self { array }
        }

        pub fn empty_connected(&self) -> impl Iterator<Item = AddIx> + '_ {
            self.array.indexed_iter().filter_map(|((x, y, z), &b)| {
                let ix = Ix3(x, y, z);
                if !b
                    && neighbor_ixs(ix)
                        .into_iter()
                        .any(|jx| self.array.get(jx).copied().unwrap_or(false))
                {
                    Some(AddIx { ix })
                } else {
                    None
                }
            })
        }
    }

    impl HasArray for Padded {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }
    }

    impl HasIx for AddIx {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }
}

mod added {
    use super::*;

    pub struct Added {
        array: Array3<bool>,
    }

    impl Added {
        pub fn from_padded(padded: &Padded, add_ixs: impl Iterator<Item = AddIx>) -> Self {
            let mut array = padded.array().clone();
            for ix in add_ixs {
                array[ix.ix()] = true;
            }
            Self { array }
        }
    }

    impl HasArray for Added {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }
    }
}

mod transformed {
    use super::*;

    #[derive(Hash, PartialEq, Eq)]
    pub struct Transformed {
        array: Array3<bool>,
    }

    impl Transformed {
        pub fn from_cropped(cropped: &Cropped) -> impl Iterator<Item = Self> + '_ {
            Permutation::all().flat_map(move |p| {
                Combination::all().filter_map(move |c| {
                    let transformed = cropped.transform(&p, &c);
                    let com = CenterOfMass::from_array(transformed.array());

                    if transformed.dim().is_valid() && com.is_valid(&transformed) {
                        Some(Self {
                            array: transformed.array().clone(),
                        })
                    } else {
                        None
                    }
                })
            })
        }
    }

    impl HasArray for Transformed {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }
    }
}

mod dim {
    use super::*;

    pub struct Dimensions {
        ix: Ix3,
    }

    impl Dimensions {
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self {
                ix: array.raw_dim(),
            }
        }

        pub fn is_valid(&self) -> bool {
            self.ix[0] >= self.ix[1] && self.ix[1] >= self.ix[2]
        }
    }

    impl HasIx for Dimensions {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }
}

mod com {
    use super::*;

    pub struct CenterOfMass {
        ix: Ix3,
    }

    impl CenterOfMass {
        // FIXME: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self {
                ix: array
                    .indexed_iter()
                    .filter_map(|((x, y, z), &b)| b.then_some(Ix3(x, y, z)))
                    .reduce(|a, b| a + b)
                    .unwrap(),
            }
        }

        pub fn is_valid(&self, cropped: &Cropped) -> bool {
            let (dx, dy, dz) = (cropped.dim().ix() - Ix3(1, 1, 1)).into_pattern();
            let (cx, cy, cz) = self.ix.into_pattern();
            let n = N::from_array(cropped.array()).num();

            if 2 * cx > n * dx || 2 * cy > n * dy || 2 * cz > n * dz {
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

    impl HasIx for CenterOfMass {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }
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

        pub fn indicies(&self) -> [usize; 3] {
            self.0
        }
    }
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

        pub fn mask(&self) -> [bool; 3] {
            self.0
        }
    }
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

        pub fn num(&self) -> usize {
            self.0
        }
    }
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

#[test]
fn test_right_answer() {
    let now = std::time::Instant::now();
    let result = solve(9);
    println!("time: {:?}", now.elapsed());

    let real = BTreeMap::from([
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 7),
        (5, 23),
        (6, 112),
        (7, 607),
        (8, 3811),
        (9, 25413),
    ]);

    assert_eq!(result, real);
}
