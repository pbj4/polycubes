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
    rustc_hash::FxHashMap,
    std::collections::BTreeMap,
    transformed::*,
};

#[allow(dead_code)]
pub fn solve_1(n: usize) -> BTreeMap<usize, usize> {
    let mut out = FxHashMap::default();

    let base = Cropped::default();

    let now = std::time::Instant::now();
    recurse_single_thread(&base, &mut out, n);
    println!("solve time: {:?}", now.elapsed());

    let mut counts = BTreeMap::new();

    for (k, v) in out {
        if v {
            *counts.entry(k.n().num()).or_default() += 1;
        }
    }

    for (k, v) in counts.iter() {
        println!("n: {:?}, count: {:?}", k, v);
    }

    counts
}

fn recurse_single_thread(cropped: &Cropped, out: &mut FxHashMap<Transformed, bool>, n: usize) {
    let canon = Transformed::from_cropped_canonical(cropped)
        .unwrap_or_else(|| Transformed::from_cropped(cropped).min().unwrap());

    let cropped = &Cropped::from_transformed(&canon);

    if let std::collections::hash_map::Entry::Vacant(entry) = out.entry(canon) {
        entry.insert(true);
    } else {
        return;
    }

    if cropped.n().num() < n {
        let padded = Padded::from_cropped(cropped);

        for add_ix in padded.empty_connected() {
            let added = Added::from_padded(&padded, std::iter::once(add_ix));
            recurse_single_thread(&Cropped::from_added(added), out, n)
        }
    }
}

#[derive(Clone, Default)]
struct FxBuildHasher {}

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = rustc_hash::FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        rustc_hash::FxHasher::default()
    }
}

const SHARD_AMOUNT: usize = 512;

#[allow(dead_code)]
pub fn solve_2(n: usize) -> BTreeMap<usize, usize> {
    let out: DashMap<(N, Dimensions), DashMap<Transformed, bool, FxBuildHasher>, FxBuildHasher> =
        DashMap::with_hasher_and_shard_amount(FxBuildHasher {}, SHARD_AMOUNT);

    let base = Cropped::default();

    let now = std::time::Instant::now();
    rayon::scope(|s| recurse_parallel(&base, &out, n, s));
    println!("solve time: {:?}", now.elapsed());

    let mut counts = BTreeMap::new();

    for ((n, _), v) in out {
        *counts.entry(n.num()).or_default() += v.len();
    }

    for (k, v) in counts.iter() {
        println!("n: {:?}, count: {:?}", k, v);
    }

    counts
}

fn recurse_parallel<'a>(
    cropped: &Cropped,
    out: &'a DashMap<(N, Dimensions), DashMap<Transformed, bool, FxBuildHasher>, FxBuildHasher>,
    n: usize,
    s: &rayon::Scope<'a>,
) {
    let canon = Transformed::from_cropped_canonical(cropped)
        .unwrap_or_else(|| Transformed::from_cropped(cropped).min().unwrap());

    let cropped = &Cropped::from_transformed(&canon);

    if let dashmap::mapref::entry::Entry::Vacant(entry) = out
        .entry((canon.n(), canon.dim()))
        .or_insert_with(|| DashMap::with_hasher_and_shard_amount(FxBuildHasher {}, SHARD_AMOUNT))
        .entry(canon)
    {
        entry.insert(true);
    } else {
        return;
    }

    if cropped.n().num() < n {
        let padded = Padded::from_cropped(cropped);

        for add_ix in padded.empty_connected() {
            let added = Added::from_padded(&padded, std::iter::once(add_ix));
            s.spawn(move |s| recurse_parallel(&Cropped::from_added(added), out, n, s));
        }
    }
}

#[allow(dead_code, clippy::type_complexity)]
pub fn solve(n: usize) -> BTreeMap<usize, usize> {
    let out: DashMap<(N, Dimensions), DashMap<Box<[u8]>, bool, FxBuildHasher>, FxBuildHasher> =
        DashMap::with_hasher_and_shard_amount(FxBuildHasher {}, SHARD_AMOUNT);

    let base = Cropped::default();

    let now = std::time::Instant::now();
    rayon::scope(|s| recurse_parallel_low_mem(&base, &out, n, s));
    println!("solve time: {:?}", now.elapsed());

    let mut counts = BTreeMap::new();

    for ((n, _), v) in out {
        *counts.entry(n.num()).or_default() += v.len();
    }

    for (k, v) in counts.iter() {
        println!("n: {:?}, count: {:?}", k, v);
    }

    counts
}

#[allow(clippy::type_complexity)]
fn recurse_parallel_low_mem<'a>(
    cropped: &Cropped,
    out: &'a DashMap<(N, Dimensions), DashMap<Box<[u8]>, bool, FxBuildHasher>, FxBuildHasher>,
    n: usize,
    s: &rayon::Scope<'a>,
) {
    let canon = Transformed::from_cropped_canonical(cropped)
        .unwrap_or_else(|| Transformed::from_cropped(cropped).min().unwrap());

    let canon_bytes = array_to_bytes(canon.array());

    let cropped = &Cropped::from_transformed(&canon);

    if let dashmap::mapref::entry::Entry::Vacant(entry) = out
        .entry((canon.n(), canon.dim()))
        .or_insert_with(|| DashMap::with_hasher_and_shard_amount(FxBuildHasher {}, SHARD_AMOUNT))
        .entry(canon_bytes)
    {
        entry.insert(true);
    } else {
        return;
    }

    if cropped.n().num() < n {
        let padded = Padded::from_cropped(cropped);

        for add_ix in padded.empty_connected() {
            let added = Added::from_padded(&padded, std::iter::once(add_ix));
            s.spawn(move |s| recurse_parallel_low_mem(&Cropped::from_added(added), out, n, s));
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

        pub fn from_transformed(transformed: &Transformed) -> Self {
            let com = transformed.com();
            let n = transformed.n();
            let array = transformed.array().clone();

            let new = Self { array, com, n };
            new.verify_com();
            new.verify_n();
            new
        }

        // ArrayBase::clone seems cheaper than ArrayView::ToOwned
        pub fn transform(self, perm: &Permutation, comb: &Combination) -> Self {
            let com = CenterOfMass::from_cropped_transform(&self, perm, comb);

            let mut array = self.array.permuted_axes(perm.indicies());
            for (i, s) in comb.mask().into_iter().enumerate() {
                if !s {
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

    #[derive(Clone)]
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
        pub fn from_padded(padded: &Padded, add_ixs: impl Iterator<Item = AddIx> + Clone) -> Self {
            let com = CenterOfMass::from_padded_to_added(padded, add_ixs.clone());
            let ix_bb = BoundingBox::from_ixs(add_ixs.clone().map(|a| a.ix()));
            let n = N::from_padded_to_added(padded, add_ixs.clone());

            let array = {
                let mut array = padded.array().clone();
                for ix in add_ixs {
                    array[ix.ix()] = true;
                }
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

    #[derive(PartialEq, Eq)]
    pub struct Transformed {
        array: Array3<bool>,
        com: CenterOfMass,
        n: N,
    }

    impl Transformed {
        pub fn from_cropped(cropped: &Cropped) -> impl Iterator<Item = Self> + '_ {
            Permutation::all()
                .filter_map(move |p| {
                    let n = cropped.n();
                    let precomp_dim = Dimensions::from_cropped_transform(cropped, &p);

                    if precomp_dim.is_valid() {
                        Some(Combination::all().filter_map(move |c| {
                            let precomp_com = CenterOfMass::from_cropped_transform(cropped, &p, &c);

                            if precomp_com.is_valid(precomp_dim, n) {
                                let transformed = cropped.clone().transform(&p, &c);
                                let com = transformed.com();
                                let n = transformed.n();

                                Some(Self {
                                    array: transformed.take_array(),
                                    com,
                                    n,
                                })
                            } else {
                                None
                            }
                        }))
                    } else {
                        None
                    }
                })
                .flatten()
        }

        pub fn from_cropped_canonical(cropped: &Cropped) -> Option<Self> {
            if cropped.dim().is_valid()
                && cropped
                    .com()
                    .is_unique_canonical(cropped.dim(), cropped.n())
            {
                Some(Self {
                    array: cropped.array().clone(),
                    com: cropped.com(),
                    n: cropped.n(),
                })
            } else {
                None
            }
        }
    }

    impl std::hash::Hash for Transformed {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.array.hash(state)
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

        pub fn from_cropped_transform(cropped: &Cropped, perm: &Permutation) -> Self {
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
            perm: &Permutation,
            comb: &Combination,
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
                if !mask[i] {
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

        pub fn from_padded_to_added(padded: &Padded, add_ixs: impl Iterator<Item = AddIx>) -> Self {
            Self {
                ix: padded.com().ix() + add_ixs.map(|a| a.ix()).reduce(|a, b| a + b).unwrap(),
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
                [true, true, true],
                [false, true, true],
                [true, false, true],
                [true, true, false],
                [false, false, true],
                [false, true, false],
                [true, false, false],
                [false, false, false],
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

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
    pub struct N(usize);

    impl N {
        // warning: O(n)
        pub fn from_array(array: &Array3<bool>) -> Self {
            Self(array.iter().filter(|&&b| b).count())
        }

        pub fn from_padded_to_added(padded: &Padded, add_ixs: impl Iterator<Item = AddIx>) -> Self {
            Self(padded.n().num() + add_ixs.count())
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

#[test]
fn test_right_answer() {
    let now = std::time::Instant::now();
    let result = solve(10);
    println!("total time: {:?}", now.elapsed());

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
        (10, 178083),
    ]);

    assert_eq!(result, real);
}
