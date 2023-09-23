use {
    added::*,
    bb::*,
    com::*,
    comb::*,
    cropped::*,
    dim::*,
    n::*,
    ndarray::{Array3, Axis, Dimension, Ix3},
    padded::*,
    perm::*,
    rustc_hash::FxHashMap,
    std::collections::{BTreeMap, HashMap},
    transformed::*,
};

#[allow(dead_code)]
pub fn solve(n: usize) -> BTreeMap<usize, usize> {
    let mut out = FxHashMap::default();

    let base = Cropped::default();

    let now = std::time::Instant::now();
    recurse_optimized_2(&base, &mut out, n);
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

#[allow(dead_code)]
fn recurse(cropped: &Cropped, out: &mut HashMap<Transformed, bool>, n: usize) {
    let mut transformations = Transformed::from_cropped(cropped);

    let canon = transformations.next().unwrap();
    let cropped = &Cropped::from_transformed(&canon);

    if let std::collections::hash_map::Entry::Vacant(entry) = out.entry(canon) {
        entry.insert(true);
    } else {
        return;
    }

    for t in transformations {
        out.entry(t).or_insert(false);
    }

    if cropped.n().num() < n {
        let padded = Padded::from_cropped(cropped);

        for add_ix in padded.empty_connected() {
            let added = Added::from_padded(&padded, std::iter::once(add_ix));
            recurse(&Cropped::from_added_optimized(added), out, n)
        }
    }
}

#[allow(dead_code)]
fn recurse_optimized(cropped: &Cropped, out: &mut FxHashMap<Transformed, bool>, n: usize) {
    let (canon, transformations) = if let Some(canon) = Transformed::from_cropped_canonical(cropped)
    {
        // happy path
        (canon, None.into_iter().flatten())
    } else {
        let mut transformations = Transformed::from_cropped_optimized(cropped);
        let canon = transformations.next().unwrap();
        (canon, Some(transformations).into_iter().flatten())
    };

    let cropped = &Cropped::from_transformed(&canon);

    if let std::collections::hash_map::Entry::Vacant(entry) = out.entry(canon) {
        entry.insert(true);
    } else {
        return;
    }

    for t in transformations {
        out.entry(t).or_insert(false);
    }

    if cropped.n().num() < n {
        let padded = Padded::from_cropped(cropped);

        for add_ix in padded.empty_connected() {
            let added = Added::from_padded(&padded, std::iter::once(add_ix));
            recurse_optimized(&Cropped::from_added_optimized(added), out, n)
        }
    }
}

fn recurse_optimized_2(cropped: &Cropped, out: &mut FxHashMap<Transformed, bool>, n: usize) {
    let canon = if let Some(canon) = Transformed::from_cropped_canonical(cropped) {
        canon
    } else {
        Transformed::from_cropped_optimized(cropped).min().unwrap()
    };

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
            recurse_optimized_2(&Cropped::from_added_optimized(added), out, n)
        }
    }
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
        // warning: O(n)
        #[allow(dead_code)]
        pub fn from_array(array: &Array3<bool>) -> Self {
            let bb = BoundingBox::from_ixs(array.indexed_iter().filter_map(|((x, y, z), &b)| {
                if b {
                    Some(Ix3(x, y, z))
                } else {
                    None
                }
            }));

            let array = bb.slice_array(array.clone());

            let com = CenterOfMass::from_array(&array);

            let n = N::from_array(&array);

            let new = Self { array, com, n };
            new.verify_com();
            new.verify_n();
            new
        }

        #[allow(dead_code)]
        pub fn from_added(added: &Added) -> Self {
            let mut bb = added.ix_bb();
            bb.add_ix(Ix3(1, 1, 1));
            bb.add_ix(added.dim().ix() - Ix3(2, 2, 2));

            let array = bb.slice_array(added.array().clone());
            let com = CenterOfMass::from_added_to_cropped(added);
            let n = added.n();

            let new = Self { array, com, n };
            new.verify_com();
            new.verify_n();
            new
        }

        pub fn from_added_optimized(added: Added) -> Self {
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

        #[allow(dead_code)]
        pub fn transform(&self, perm: &Permutation, comb: &Combination) -> Self {
            let com = CenterOfMass::from_cropped_transform(self, perm, comb);

            let mut view = self.array.view().permuted_axes(perm.indicies());
            for (i, s) in comb.mask().into_iter().enumerate() {
                if !s {
                    view.invert_axis(Axis(i))
                }
            }
            let array = view.to_owned();

            let new = Self {
                array,
                com,
                n: self.n,
            };
            new.verify_com();
            new.verify_n();
            new
        }

        // ArrayBase::clone seems cheaper than ArrayView::ToOwned
        pub fn transform_optimized(self, perm: &Permutation, comb: &Combination) -> Self {
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
            Permutation::all().flat_map(move |p| {
                Combination::all().filter_map(move |c| {
                    let n = cropped.n();

                    let precomp_dim = Dimensions::from_cropped_transform(cropped, &p);
                    let precomp_com = CenterOfMass::from_cropped_transform(cropped, &p, &c);
                    let precomp_is_valid =
                        precomp_dim.is_valid() && precomp_com.is_valid(precomp_dim, n);

                    if !precomp_is_valid {
                        None
                    } else {
                        let transformed = cropped.clone().transform_optimized(&p, &c);
                        let com = transformed.com();
                        let n = transformed.n();

                        Some(Self {
                            array: transformed.take_array(),
                            com,
                            n,
                        })
                    }
                })
            })
        }

        pub fn from_cropped_optimized(cropped: &Cropped) -> impl Iterator<Item = Self> + '_ {
            Permutation::all()
                .filter_map(move |p| {
                    let n = cropped.n();
                    let precomp_dim = Dimensions::from_cropped_transform(cropped, &p);

                    if precomp_dim.is_valid() {
                        Some(Combination::all().filter_map(move |c| {
                            let precomp_com = CenterOfMass::from_cropped_transform(cropped, &p, &c);

                            if precomp_com.is_valid(precomp_dim, n) {
                                let transformed = cropped.clone().transform_optimized(&p, &c);
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
            self.com
                .cmp(&other.com)
                .then_with(|| self.array.iter().cmp(other.array.iter()))
                .then_with(|| self.n.cmp(&other.n))
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

    #[derive(PartialEq, Debug, Clone, Copy)]
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
}

mod com {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
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
        (10, 178083),
    ]);

    assert_eq!(result, real);
}
