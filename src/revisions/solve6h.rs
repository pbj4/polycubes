use {
    added::*,
    bb::*,
    com::*,
    comb::*,
    cropped::*,
    dashmap::DashMap,
    dim::*,
    n::*,
    ndarray::{Array3, Axis, Dimension, Ix3},
    padded::*,
    perm::*,
    rayon::prelude::*,
    std::collections::BTreeMap,
    std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    transformed::*,
};

#[derive(Clone, Default)]
struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = rustc_hash::FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        rustc_hash::FxHasher::default()
    }
}

const SHARD_AMOUNT: usize = 512;

#[allow(dead_code, clippy::type_complexity)]
pub fn solve(n: usize) -> BTreeMap<usize, (usize, usize)> {
    let out: DashMap<(N, Dimensions), DashMap<Box<[u8]>, bool, FxBuildHasher>, FxBuildHasher> =
        DashMap::with_hasher_and_shard_amount(FxBuildHasher, SHARD_AMOUNT);

    let base = Cropped::default();

    let now = std::time::Instant::now();
    rayon::scope(|s| recurse_parallel_rotations(&base, &out, n, s));
    println!("solve time: {:?}", now.elapsed());

    let counts = {
        let mut counts: BTreeMap<usize, (AtomicUsize, AtomicUsize)> = BTreeMap::new();
        for i in 1..=n {
            counts.insert(i, Default::default());
        }
        Arc::new(counts)
    };

    out.into_iter().par_bridge().for_each(|((n, _), m)| {
        let (r, s) = counts.get(&n.num()).unwrap();
        for (_, b) in m {
            if b {
                s.fetch_add(1, Ordering::Relaxed);
            } else {
                r.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    let counts = Arc::into_inner(counts)
        .unwrap()
        .into_iter()
        .map(|(n, (r, p))| {
            let r = r.into_inner();
            let p = p.into_inner();
            (n, (p + 2 * r, p + r))
        })
        .collect::<BTreeMap<_, _>>();

    for (k, (r, s)) in counts.iter() {
        println!("n: {:?}, r: {:?}, p: {:?}", k, r, s);
    }

    counts
}

#[allow(clippy::type_complexity)]
fn recurse_parallel_rotations<'a>(
    cropped: &Cropped,
    out: &'a DashMap<(N, Dimensions), DashMap<Box<[u8]>, bool, FxBuildHasher>, FxBuildHasher>,
    n: usize,
    s: &rayon::Scope<'a>,
) {
    let (canon, has_reflection_symmetry) = Transformed::from_cropped_canonical_only(cropped);

    let canon_bytes = array_to_bytes(canon.array());

    if let dashmap::mapref::entry::Entry::Vacant(entry) = out
        .entry((canon.n(), canon.dim()))
        .or_insert_with(|| DashMap::with_hasher_and_shard_amount(FxBuildHasher, SHARD_AMOUNT))
        .entry(canon_bytes)
    {
        entry.insert(has_reflection_symmetry);
    } else {
        return;
    }

    let cropped = &Cropped::from_transformed(canon);

    if cropped.n().num() < n {
        let padded = Padded::from_cropped(cropped);

        for add_ix in padded.empty_connected() {
            let added = Added::from_padded(&padded, add_ix);
            s.spawn(move |s| recurse_parallel_rotations(&Cropped::from_added(added), out, n, s));
        }
    }
}

fn array_to_bytes(array: &Array3<bool>) -> Box<[u8]> {
    let mut buffer = vec![0u8; (array.len() - 1) / 8 + 1];
    for (i, &b) in array.iter().enumerate() {
        buffer[i >> 3] |= if b { 1 } else { 0 } << (i & 0b111);
    }
    buffer.into()
}

trait HasArray {
    fn array(&self) -> &Array3<bool>;

    fn take_array(self) -> Array3<bool>;

    fn dim(&self) -> Dimensions {
        Dimensions::from_array(self.array())
    }
}

trait HasCom: HasArray {
    fn com(&self) -> CenterOfMass;

    fn verify_com(&self) {
        debug_assert_eq!(CenterOfMass::from_array(self.array()), self.com());
    }
}

trait HasN: HasArray {
    fn n(&self) -> N;

    fn verify_n(&self) {
        debug_assert_eq!(N::from_array(self.array()), self.n());
    }
}

trait HasIx {
    fn ix(&self) -> Ix3;
}

mod cropped {
    use super::*;

    #[derive(Clone)]
    pub struct Cropped {
        array: Array3<bool>,
        com: CenterOfMass,
        n: N,
    }

    impl Cropped {
        pub fn from_added(added: Added) -> Self {
            let mut bb = added.ix_bb();
            bb.add_ix(Ix3(1, 1, 1));
            bb.add_ix(added.dim().ix() - Ix3(2, 2, 2));

            let com = CenterOfMass::from_added_to_cropped(&added);
            let n = added.n();
            let array = bb.slice_array(added.take_array());

            let new = Self { array, com, n };
            new.verify_com();
            new.verify_n();
            new
        }

        pub fn from_transformed(transformed: Transformed) -> Self {
            let com = transformed.com();
            let n = transformed.n();
            let array = transformed.take_array();

            let new = Self { array, com, n };
            new.verify_com();
            new.verify_n();
            new
        }

        pub fn transform(self, perm: Permutation, comb: Combination) -> Self {
            let com = CenterOfMass::from_cropped_transform(&self, perm, comb);

            let mut array = self.array.permuted_axes(perm.indicies());
            for (i, s) in comb.mask().into_iter().enumerate() {
                if s {
                    array.invert_axis(Axis(i))
                }
            }

            let new = Self {
                array,
                com,
                n: self.n,
            };
            new.verify_com();
            new.verify_n();
            new
        }
    }

    impl Default for Cropped {
        fn default() -> Self {
            let array = Array3::from_elem((1, 1, 1), true);
            let com = CenterOfMass::from_array(&array);
            let n = N::from_array(&array);
            Self { array, com, n }
        }
    }

    impl HasArray for Cropped {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }

        fn take_array(self) -> Array3<bool> {
            self.array
        }
    }

    impl HasCom for Cropped {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl HasN for Cropped {
        fn n(&self) -> N {
            self.n
        }
    }
}

mod padded {
    use super::*;

    #[derive(Clone)]
    pub struct Padded {
        array: Array3<bool>,
        com: CenterOfMass,
        n: N,
    }

    #[derive(Clone, Copy)]
    pub struct AddIx {
        ix: Ix3,
    }

    impl Padded {
        pub fn from_cropped(cropped: &Cropped) -> Self {
            let com = CenterOfMass::from_cropped_to_padded(cropped);

            let array = {
                let mut array = Array3::from_elem(cropped.dim().ix() + Ix3(2, 2, 2), false);
                #[allow(clippy::reversed_empty_ranges)]
                let mut slice = array.slice_mut(ndarray::s![1..-1, 1..-1, 1..-1]);
                slice.assign(cropped.array());
                array
            };

            let new = Self {
                array,
                com,
                n: cropped.n(),
            };
            new.verify_com();
            new.verify_n();
            new
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

        fn take_array(self) -> Array3<bool> {
            self.array
        }
    }

    impl HasCom for Padded {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl HasN for Padded {
        fn n(&self) -> N {
            self.n
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
        com: CenterOfMass,
        ix_bb: BoundingBox,
        n: N,
    }

    impl Added {
        pub fn from_padded(padded: &Padded, add_ix: AddIx) -> Self {
            let com = CenterOfMass::from_padded_to_added(padded, add_ix);
            let ix_bb = BoundingBox::new(add_ix.ix());
            let n = N::from_padded_to_added(padded, add_ix);

            let array = {
                let mut array = padded.array().clone();
                array[add_ix.ix()] = true;
                array
            };

            let new = Self {
                array,
                com,
                ix_bb,
                n,
            };
            new.verify_com();
            new.verify_n();
            new
        }

        pub fn ix_bb(&self) -> BoundingBox {
            self.ix_bb
        }
    }

    impl HasArray for Added {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }

        fn take_array(self) -> Array3<bool> {
            self.array
        }
    }

    impl HasCom for Added {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl HasN for Added {
        fn n(&self) -> N {
            self.n
        }
    }
}

mod transformed {
    use super::*;

    #[derive(Clone)]
    pub struct Transformed {
        com: CenterOfMass,
        n: N,
        array: Array3<bool>,
    }

    impl Transformed {
        pub fn from_cropped_canonical_only(cropped: &Cropped) -> (Self, bool) {
            let mut lowest: Option<Self> = None;
            let mut first: Option<(Self, bool)> = None;
            let mut has_reflection_symmetry = false;

            let n = cropped.n();

            for p in Permutation::all() {
                let precomp_dim = Dimensions::from_cropped_transform(cropped, p);

                if precomp_dim.is_valid() {
                    for c in Combination::all() {
                        let precomp_com = CenterOfMass::from_cropped_transform(cropped, p, c);

                        if precomp_com.is_valid(precomp_dim, n) {
                            let new = Self {
                                array: cropped.clone().transform(p, c).take_array(),
                                com: precomp_com,
                                n,
                            };

                            let chirality = is_reflected(p, c);

                            if let Some((first_cube, first_chirality)) = &first {
                                if !has_reflection_symmetry
                                    && &chirality != first_chirality
                                    && &new == first_cube
                                {
                                    has_reflection_symmetry = true;
                                }
                            } else {
                                first = Some((new.clone(), chirality));
                            }

                            if let Some(prev_lowest) = lowest {
                                lowest = Some(prev_lowest.min(new));
                            } else {
                                lowest = Some(new);
                            }
                        }
                    }
                }
            }

            (lowest.unwrap(), has_reflection_symmetry)
        }
    }

    impl PartialOrd for Transformed {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Transformed {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.n
                .cmp(&other.n)
                .then_with(|| self.com.cmp(&other.com))
                .then_with(|| self.array.iter().cmp(other.array.iter()))
        }
    }

    impl PartialEq for Transformed {
        fn eq(&self, other: &Self) -> bool {
            self.n.eq(&other.n) && self.com.eq(&other.com) && self.array().eq(other.array())
        }
    }

    impl Eq for Transformed {}

    impl HasArray for Transformed {
        fn array(&self) -> &Array3<bool> {
            &self.array
        }

        fn take_array(self) -> Array3<bool> {
            self.array
        }
    }

    impl HasCom for Transformed {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl HasN for Transformed {
        fn n(&self) -> N {
            self.n
        }
    }
}

mod dim {
    use super::*;

    #[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
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

        pub fn from_cropped_transform(cropped: &Cropped, perm: Permutation) -> Self {
            let dim = cropped.dim().ix();
            let [x, y, z] = perm.indicies();
            Self {
                ix: Ix3(dim[x], dim[y], dim[z]),
            }
        }
    }

    impl HasIx for Dimensions {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }

    impl PartialOrd for Dimensions {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Dimensions {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.ix.into_pattern().cmp(&other.ix.into_pattern())
        }
    }
}

mod com {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CenterOfMass {
        ix: Ix3,
    }

    impl CenterOfMass {
        // warning: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self {
                ix: array
                    .indexed_iter()
                    .filter_map(|((x, y, z), &b)| b.then_some(Ix3(x, y, z)))
                    .reduce(|a, b| a + b)
                    .unwrap(),
            }
        }

        // self and dim MUST have matching transformations (no rotating one and not the other)
        pub fn is_valid(&self, dim: Dimensions, n: N) -> bool {
            let (dx, dy, dz) = (dim.ix() - Ix3(1, 1, 1)).into_pattern();
            let (cx, cy, cz) = self.ix.into_pattern();
            let n = n.num();

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

        #[allow(dead_code)]
        pub fn is_unique_canonical(&self, dim: Dimensions, n: N) -> bool {
            let (dx, dy, dz) = (dim.ix() - Ix3(1, 1, 1)).into_pattern();
            let (cx, cy, cz) = self.ix.into_pattern();
            let n = n.num();

            if 2 * cx >= n * dx || 2 * cy >= n * dy || 2 * cz >= n * dz {
                return false;
            }

            let xy = cy * dx < dy * cx;
            let yz = cz * dy < dz * cy;

            match (dx == dy, dy == dz) {
                (false, false) => true,
                (true, false) => xy,
                (false, true) => yz,
                (true, true) => xy && yz,
            }
        }

        pub fn from_cropped_transform(
            cropped: &Cropped,
            perm: Permutation,
            comb: Combination,
        ) -> Self {
            let mut com = {
                let [x, y, z] = perm.indicies();
                let com = cropped.com().ix();
                [com[x], com[y], com[z]]
            };

            let dim: [usize; 3] = Dimensions::from_cropped_transform(cropped, perm)
                .ix()
                .into_pattern()
                .into();

            let n = cropped.n().num();

            let mask = comb.mask();

            for i in 0..3 {
                if mask[i] {
                    com[i] = (dim[i] - 1) * n - com[i];
                }
            }

            {
                let [x, y, z] = com;
                Self { ix: Ix3(x, y, z) }
            }
        }

        pub fn from_cropped_to_padded(cropped: &Cropped) -> Self {
            Self {
                ix: cropped.com().ix() + Ix3(1, 1, 1) * cropped.n().num(),
            }
        }

        pub fn from_padded_to_added(padded: &Padded, add_ix: AddIx) -> Self {
            Self {
                ix: padded.com().ix() + add_ix.ix(),
            }
        }

        pub fn from_added_to_cropped(added: &Added) -> Self {
            let mut bb = added.ix_bb();
            bb.add_ix(Ix3(1, 1, 1));
            Self {
                ix: added.com().ix() - bb.min() * added.n().num(),
            }
        }
    }

    impl HasIx for CenterOfMass {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }

    impl PartialOrd for CenterOfMass {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CenterOfMass {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.ix.into_pattern().cmp(&other.ix.into_pattern())
        }
    }
}

mod perm {
    #[derive(Clone, Copy)]
    pub struct Permutation([usize; 3]);

    impl Permutation {
        pub fn all() -> [Self; 6] {
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 0, 1],
                [0, 2, 1],
                [1, 2, 0],
                [2, 1, 0],
            ]
            //.into_iter()
            .map(Self)
        }

        pub fn indicies(&self) -> [usize; 3] {
            self.0
        }
    }
}

mod comb {
    // true represents inversion
    #[derive(Clone, Copy)]
    pub struct Combination([bool; 3]);

    impl Combination {
        pub fn all() -> [Self; 8] {
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
            .map(Self)
        }

        pub fn mask(&self) -> [bool; 3] {
            self.0
        }
    }
}

mod n {
    use super::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
    pub struct N(usize);

    impl N {
        // warning: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self(array.iter().filter(|&&b| b).count())
        }

        pub fn from_padded_to_added(padded: &Padded, _add_ix: AddIx) -> Self {
            Self(padded.n().num() + 1)
        }

        pub fn num(&self) -> usize {
            self.0
        }
    }
}

mod bb {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct BoundingBox {
        min: [usize; 3],
        max: [usize; 3],
    }

    impl BoundingBox {
        pub fn new(ix: Ix3) -> Self {
            Self {
                min: ix.into_pattern().into(),
                max: ix.into_pattern().into(),
            }
        }

        #[allow(dead_code)]
        pub fn from_ixs(mut ixs: impl Iterator<Item = Ix3>) -> Self {
            let mut new = Self::new(ixs.next().unwrap());

            for ix in ixs {
                new.add_ix(ix);
            }

            new
        }

        pub fn add_ix(&mut self, ix: Ix3) {
            let ix: [_; 3] = ix.into_pattern().into();
            for (i, ix) in ix.into_iter().enumerate() {
                self.min[i] = self.min[i].min(ix);
                self.max[i] = self.max[i].max(ix);
            }
        }

        pub fn slice_array(&self, array: Array3<bool>) -> Array3<bool> {
            let [xmin, ymin, zmin] = self.min;
            let [xmax, ymax, zmax] = self.max;
            array.slice_move(ndarray::s![xmin..=xmax, ymin..=ymax, zmin..=zmax])
        }

        pub fn min(&self) -> Ix3 {
            Ix3(self.min[0], self.min[1], self.min[2])
        }

        #[allow(dead_code)]
        pub fn max(&self) -> Ix3 {
            Ix3(self.max[0], self.max[1], self.max[2])
        }
    }
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

fn is_reflected(perm: Permutation, comb: Combination) -> bool {
    let comb = comb.mask();

    // shift x to first position (valid rotation)
    let mut perm = perm.indicies();
    let x_pos = perm.iter().position(|&p| p == 0).unwrap();
    perm.rotate_left(x_pos);

    // check for odd number of reflections
    // y and z being swapped and axis inversions are reflections
    (perm[1] != 1) ^ comb[0] ^ comb[1] ^ comb[2]
}

#[test]
fn test_small_n() {
    let now = std::time::Instant::now();
    let result = solve(10);
    println!("total time: {:?}", now.elapsed());

    let real = BTreeMap::from([
        (1, (1, 1)),
        (2, (1, 1)),
        (3, (2, 2)),
        (4, (8, 7)),
        (5, (29, 23)),
        (6, (166, 112)),
        (7, (1023, 607)),
        (8, (6922, 3811)),
        (9, (48311, 25413)),
        (10, (346543, 178083)),
    ]);

    assert_eq!(result, real);
}
