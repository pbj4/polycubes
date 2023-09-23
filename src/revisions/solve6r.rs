use {
    added::*,
    ap::*,
    bb::*,
    com::*,
    comb::*,
    cropped::*,
    dim::*,
    n::*,
    ndarray::{Array3, ArrayView3, Axis, Dimension, Ix3},
    padded::*,
    perm::*,
    std::{
        cell::RefCell,
        cmp::Ordering,
        collections::{BTreeMap, HashMap},
        sync::atomic::AtomicUsize,
    },
    transformed::*,
};

#[allow(dead_code)]
pub fn solve(n: usize) -> BTreeMap<usize, (usize, usize)> {
    let (base, mirror_sym) = Transformed::from_cropped_canonical(&Cropped::default());
    let out = BTreeMap::from_iter((1..=n).map(|n| (n, Default::default())));

    rayon::scope(|s| recurse_hashless_min_point(&base, mirror_sym, &out, n, s));

    let counts: BTreeMap<usize, (usize, usize)> = out
        .into_iter()
        .map(|(n, (r, p))| {
            let (r, p) = (r.into_inner(), p.into_inner());
            (n, (p + 2 * r, p + r))
        })
        .collect();

    for (k, (r, p)) in counts.iter() {
        println!("n: {:?}, r: {:?}, p: {:?}", k, r, p);
    }

    counts
}

fn recurse_hashless_min_point<'a>(
    transformed: &Transformed,
    mirror_symmetry: bool,
    out: &'a BTreeMap<usize, (AtomicUsize, AtomicUsize)>,
    target_n: usize,
    s: &rayon::Scope<'a>,
) {
    let current_n = transformed.n();

    let (r, p) = out.get(&current_n.num()).unwrap();
    if mirror_symmetry {
        p.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        r.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    if current_n.num() >= target_n {
        return;
    }

    let padded = Padded::from_cropped(&Cropped::from_transformed(*transformed));
    let padded_aps = ArticulationPoints::from_view(padded.array_view());

    let all_added = {
        let mut vec = Vec::with_capacity(current_n.num().pow(2));
        vec.extend(
            padded
                .list_addixs()
                .map(|add_ix| (Added::from_padded(&padded, add_ix), add_ix)),
        );
        vec
    };

    let unique_addeds: HashMap<_, _, FxBuildHasher> = all_added
        .iter()
        .map(|(added, add_ix)| {
            let cropped = Cropped::from_added(added);
            let cropped_aps = cropped.bb_used().slice_view(padded_aps.ap_view());
            let add_ix = add_ix.ix() - cropped.bb_used().min();
            let (t, t_add_ix) = Transformed::from_cropped_min_point(&cropped, add_ix);
            let transformed_aps = t.transform_used().transform_view(cropped_aps);
            ((t, t_add_ix), (added, transformed_aps))
        })
        .collect();

    'added: for ((added_transformed, add_ix), (added, transformed_aps)) in unique_addeds {
        let added_transformed_view = added_transformed.array_view();

        let addix_on_bridge = neighbor_ixs(add_ix)
            .into_iter()
            .filter(|&ix| *added_transformed_view.get(ix).unwrap_or(&false))
            .count()
            == 1
            && added_transformed.n().num() > 2;

        let possible_smaller = {
            let mut vec = Vec::with_capacity(added_transformed_view.len() / 2);
            vec.extend(
                added_transformed_view
                    .indexed_iter()
                    .filter_map(|(ix, b)| {
                        if *b {
                            Some(usize3_to_ix(ix.into()))
                        } else {
                            None
                        }
                    })
                    .take_while(|ix| ix.into_pattern() < add_ix.into_pattern()),
            );
            vec
        };

        if possible_smaller
            .iter()
            .any(|&ix| !((addix_on_bridge && is_neighbor(ix, add_ix)) || transformed_aps[ix]))
            || possible_smaller.iter().any(|&ix| {
                ArticulationPoints::can_remove_one(
                    added_transformed_view,
                    added_transformed.n(),
                    ix,
                )
            })
        {
            continue 'added;
        }

        let added = added.clone();
        s.spawn(move |s| {
            let (canonical_added, mirror_symmetry) =
                Transformed::from_cropped_canonical(&Cropped::from_added(&added));
            recurse_hashless_min_point(&canonical_added, mirror_symmetry, out, target_n, s);
        })
    }
}

#[derive(Clone, Default)]
struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = rustc_hash::FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        rustc_hash::FxHasher::default()
    }
}

trait HasView<'a, 'b> {
    fn array_view(&'b self) -> ArrayView3<'a, bool>;

    fn dim(&'b self) -> Dimensions {
        Dimensions::from_view(self.array_view())
    }
}

trait HasArray<'a>: HasView<'a, 'a> {
    fn take_array(self) -> Array3<bool>;
}

trait HasCom<'a, 'b>: HasView<'a, 'b> {
    fn com(&self) -> CenterOfMass;

    fn verify_com(&'b self) {
        debug_assert_eq!(CenterOfMass::from_view(self.array_view()), self.com());
    }
}

trait HasN<'a, 'b>: HasView<'a, 'b> {
    fn n(&self) -> N;

    fn verify_n(&'b self) {
        debug_assert_eq!(N::from_view(self.array_view()), self.n());
    }
}

trait HasComCC<'a, 'b>: HasN<'a, 'b> {
    fn com_cc(&self) -> CenterOfMassCalcCache;

    fn verify_com_cc(&'b self) {
        debug_assert_eq!(
            CenterOfMassCalcCache::new(self.dim(), self.n()),
            self.com_cc()
        );
    }
}

trait HasIx {
    fn ix(&self) -> Ix3;
}

mod cropped {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct Cropped<'a> {
        view: ArrayView3<'a, bool>,
        com: CenterOfMass,
        n: N,
        com_cc: CenterOfMassCalcCache,
        bb: BoundingBox,
    }

    impl<'a> Cropped<'a> {
        pub fn from_added(added: &'a Added) -> Self {
            let mut bb = BoundingBox::new(added.add_ix().ix());
            bb.add_ix(Ix3(1, 1, 1));
            bb.add_ix(added.dim().ix() - Ix3(2, 2, 2));

            let com = CenterOfMass::from_added_to_cropped(added);
            let n = added.n();
            let view = bb.slice_view(added.array_view());
            let com_cc = CenterOfMassCalcCache::new(Dimensions::from_view(view), n);

            let new = Self {
                view,
                com,
                n,
                com_cc,
                bb,
            };
            new.verify_com();
            new.verify_n();
            new.verify_com_cc();
            new
        }

        pub fn from_transformed(transformed: Transformed<'a>) -> Self {
            let com = transformed.com();
            let n = transformed.n();
            let view = transformed.array_view();
            let com_cc = CenterOfMassCalcCache::new(Dimensions::from_view(view), n);

            let new = Self {
                view,
                com,
                n,
                com_cc,
                bb: BoundingBox::new_min_max(Ix3(0, 0, 0), transformed.dim().ix() - Ix3(1, 1, 1)),
            };
            new.verify_com();
            new.verify_n();
            new.verify_com_cc();
            new
        }

        pub fn bb_used(&self) -> BoundingBox {
            self.bb
        }
    }

    impl<'a> Default for Cropped<'a> {
        fn default() -> Self {
            let view = ArrayView3::from_shape((1, 1, 1), &[true; 1]).unwrap();
            let com = CenterOfMass::from_view(view);
            let n = N::from_view(view);
            let com_cc = CenterOfMassCalcCache::new(Dimensions::from_view(view), n);
            Self {
                view,
                com,
                n,
                com_cc,
                bb: BoundingBox::new(Ix3(0, 0, 0)),
            }
        }
    }

    impl<'a, 'b> HasView<'a, 'b> for Cropped<'a> {
        fn array_view(&self) -> ArrayView3<'a, bool> {
            self.view
        }
    }

    impl<'a, 'b> HasCom<'a, 'b> for Cropped<'a> {
        #[inline]
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl<'a, 'b> HasN<'a, 'b> for Cropped<'a> {
        fn n(&self) -> N {
            self.n
        }
    }

    impl<'a, 'b> HasComCC<'a, 'b> for Cropped<'a> {
        #[inline]
        fn com_cc(&self) -> CenterOfMassCalcCache {
            self.com_cc
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

    #[derive(Clone, Copy, Debug)]
    pub struct AddIx {
        ix: Ix3,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct RemoveIx {
        ix: Ix3,
    }

    impl Padded {
        pub fn from_cropped(cropped: &Cropped) -> Self {
            let com = CenterOfMass::from_cropped_to_padded(cropped);

            let array = {
                let mut array = Array3::from_elem(cropped.dim().ix() + Ix3(2, 2, 2), false);
                #[allow(clippy::reversed_empty_ranges)]
                let slice = array.slice_mut(ndarray::s![1..-1, 1..-1, 1..-1]);
                cropped.array_view().assign_to(slice);
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

        pub fn list_addixs(&self) -> impl Iterator<Item = AddIx> + '_ {
            self.array.indexed_iter().filter_map(|(ix, &b)| {
                let ix = usize3_to_ix(ix.into());
                if !b
                    && neighbor_ixs(ix)
                        .into_iter()
                        .any(|jx| *self.array.get(jx).unwrap_or(&false))
                {
                    Some(AddIx { ix })
                } else {
                    None
                }
            })
        }
    }

    impl<'a> HasView<'a, 'a> for Padded {
        fn array_view(&self) -> ArrayView3<bool> {
            self.array.view()
        }
    }

    impl<'a> HasArray<'a> for Padded {
        fn take_array(self) -> Array3<bool> {
            self.array
        }
    }

    impl<'a> HasCom<'a, 'a> for Padded {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl<'a> HasN<'a, 'a> for Padded {
        fn n(&self) -> N {
            self.n
        }
    }

    impl HasIx for AddIx {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }

    impl HasIx for RemoveIx {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }
}

mod added {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Added {
        array: Array3<bool>,
        com: CenterOfMass,
        add_ix: AddIx,
        n: N,
    }

    impl Added {
        pub fn from_padded(padded: &Padded, add_ix: AddIx) -> Self {
            let com = CenterOfMass::from_padded_to_added(padded, add_ix);
            let n = N::from_padded_to_added(padded, add_ix);

            let array = {
                let mut array = padded.array_view().to_owned();
                array[add_ix.ix()] = true;
                array
            };

            let new = Self {
                array,
                com,
                add_ix,
                n,
            };
            new.verify_com();
            new.verify_n();
            new
        }

        pub fn add_ix(&self) -> AddIx {
            self.add_ix
        }
    }

    impl<'a> HasView<'a, 'a> for Added {
        fn array_view(&self) -> ArrayView3<bool> {
            self.array.view()
        }
    }

    impl<'a> HasArray<'a> for Added {
        fn take_array(self) -> Array3<bool> {
            self.array
        }
    }

    impl<'a> HasCom<'a, 'a> for Added {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl<'a> HasN<'a, 'a> for Added {
        fn n(&self) -> N {
            self.n
        }
    }
}

mod transformed {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct Transformed<'a> {
        n: N,
        com: CenterOfMass,
        view: ArrayView3<'a, bool>,

        // not checked for canonicity
        transform: Transformation,
    }

    // does inversion first, then permutation
    #[derive(Clone, Copy, Debug)]
    pub struct Transformation {
        comb: Combination,
        perm: Permutation,
    }

    impl<'a> Transformed<'a> {
        pub fn from_cropped_canonical(cropped: &Cropped<'a>) -> (Self, bool) {
            let mut lowest: Option<Self> = None;
            let mut first: Option<(Self, bool)> = None;
            let mut has_reflection_symmetry = false;

            let n = cropped.n();

            let valid_combs = Combination::all_valid_octant(cropped.com(), cropped.com_cc());
            let valid_octal_com = cropped
                .com()
                .invert(*valid_combs.first().unwrap(), cropped.com_cc());

            let valid_perms = Permutation::all_valid(valid_octal_com, cropped.dim());
            let valid_com = valid_octal_com.permute(*valid_perms.first().unwrap());

            // fast path
            if let ([comb], [perm]) = (valid_combs.as_slice(), valid_perms.as_slice()) {
                let transform = Transformation {
                    comb: *comb,
                    perm: *perm,
                };
                return (
                    Self {
                        view: transform.transform_view(cropped.array_view()),
                        com: valid_com,
                        n,
                        transform,
                    },
                    false,
                );
            }

            for comb in valid_combs {
                for perm in valid_perms.iter().copied() {
                    let transform = Transformation { comb, perm };

                    let new = Self {
                        view: transform.transform_view(cropped.array_view()),
                        com: valid_com,
                        n,
                        transform,
                    };

                    let chirality = is_reflected(perm, comb);

                    if let Some((first_cube, first_chirality)) = &first {
                        if !has_reflection_symmetry
                            && &chirality != first_chirality
                            && &new == first_cube
                        {
                            has_reflection_symmetry = true;
                        }
                    } else {
                        first = Some((new, chirality));
                    }

                    if let Some(prev_lowest) = lowest {
                        lowest = Some(prev_lowest.min(new));
                    } else {
                        lowest = Some(new);
                    }
                }
            }

            let new = lowest.unwrap();
            new.verify_com();
            new.verify_n();
            (new, has_reflection_symmetry)
        }

        pub fn from_cropped_min_point(cropped: &Cropped<'a>, ix: Ix3) -> (Self, Ix3) {
            let mut lowest: Option<(Self, [usize; 3])> = None;

            let n = cropped.n();

            let valid_combs = Combination::all_valid_octant(cropped.com(), cropped.com_cc());
            let valid_octal_com = cropped
                .com()
                .invert(*valid_combs.first().unwrap(), cropped.com_cc());

            let valid_perms = Permutation::all_valid(valid_octal_com, cropped.dim());
            let valid_com = valid_octal_com.permute(*valid_perms.first().unwrap());

            // fast path
            if let ([comb], [perm]) = (valid_combs.as_slice(), valid_perms.as_slice()) {
                let transform = Transformation {
                    comb: *comb,
                    perm: *perm,
                };

                return (
                    Self {
                        view: transform.transform_view(cropped.array_view()),
                        com: valid_com,
                        n,
                        transform,
                    },
                    perm.permute_ix(comb.invert_ix(ix, cropped.dim().ix() - Ix3(1, 1, 1))),
                );
            }

            for comb in valid_combs {
                for perm in valid_perms.iter().copied() {
                    let transform = Transformation { comb, perm };

                    let new = Self {
                        view: transform.transform_view(cropped.array_view()),
                        com: valid_com,
                        n,
                        transform,
                    };

                    let transformed_ix = perm
                        .permute_ix(comb.invert_ix(ix, cropped.dim().ix() - Ix3(1, 1, 1)))
                        .into_pattern()
                        .into();

                    if let Some(prev_lowest) = lowest {
                        lowest = Some(prev_lowest.min((new, transformed_ix)));
                    } else {
                        lowest = Some((new, transformed_ix));
                    }
                }
            }

            let (new, transformed_ix) = lowest.unwrap();
            new.verify_com();
            new.verify_n();
            (new, usize3_to_ix(transformed_ix))
        }

        pub fn cmp_with_lifetime(&self, other: &Transformed) -> Ordering {
            self.com
                .cmp(&other.com)
                .then_with(|| self.view.iter().cmp(other.view.iter()))
        }

        pub fn transform_used(&self) -> Transformation {
            self.transform
        }
    }

    impl Transformation {
        pub fn transform_view<'a>(&self, view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            self.perm.permute_view(self.comb.invert_view(view))
        }
    }

    impl<'a> PartialOrd for Transformed<'a> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a> Ord for Transformed<'a> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.cmp_with_lifetime(other)
        }
    }

    impl<'a, 'b> PartialEq<Transformed<'b>> for Transformed<'a> {
        fn eq(&self, other: &Transformed<'b>) -> bool {
            self.com.eq(&other.com) && self.array_view().eq(&other.array_view())
        }
    }

    impl<'a> Eq for Transformed<'a> {}

    impl<'a> std::hash::Hash for Transformed<'a> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.view.hash(state);
        }
    }

    impl<'a, 'b> HasView<'a, 'b> for Transformed<'a> {
        fn array_view(&self) -> ArrayView3<'a, bool> {
            self.view
        }
    }

    impl<'a, 'b> HasCom<'a, 'b> for Transformed<'a> {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl<'a, 'b> HasN<'a, 'b> for Transformed<'a> {
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
        pub fn from_view(view: ArrayView3<bool>) -> Self {
            Self { ix: view.raw_dim() }
        }

        #[allow(dead_code)]
        pub fn is_valid(&self) -> bool {
            self.ix[0] <= self.ix[1] && self.ix[1] <= self.ix[2]
        }
    }

    impl HasIx for Dimensions {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }

    impl PartialOrd for Dimensions {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Dimensions {
        fn cmp(&self, other: &Self) -> Ordering {
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

        pub fn from_view(view: ArrayView3<bool>) -> Self {
            Self {
                ix: view
                    .indexed_iter()
                    .filter_map(|(ix, &b)| b.then_some(usize3_to_ix(ix.into())))
                    .reduce(|a, b| a + b)
                    .unwrap(),
            }
        }

        // self and dim MUST have matching transformations (no rotating one and not the other)
        #[allow(dead_code)]
        pub fn is_valid(&self, com_cc: CenterOfMassCalcCache) -> bool {
            if !self.octant_is_valid(com_cc) {
                return false;
            }

            self.diag_is_valid(com_cc)
        }

        pub fn octant_is_valid(&self, com_cc: CenterOfMassCalcCache) -> bool {
            let [x, y, z] = self.classify_octant(com_cc);
            x.is_le() && y.is_le() && z.is_le()
        }

        pub fn classify_octant(&self, com_cc: CenterOfMassCalcCache) -> [Ordering; 3] {
            ix_cmp_elementwise(self.ix * 2, com_cc.dn)
        }

        pub fn diag_is_valid(&self, com_cc: CenterOfMassCalcCache) -> bool {
            let (cx, cy, cz) = self.ix.into_pattern();
            let (dx, dy, dz) = com_cc.dn.into_pattern();

            let xy = dy * cx <= cy * dx;
            let yz = dz * cy <= cz * dy;

            match (dx == dy, dy == dz) {
                (false, false) => true,
                (true, false) => xy,
                (false, true) => yz,
                (true, true) => xy && yz,
            }
        }

        #[inline]
        pub fn permute(&self, perm: Permutation) -> Self {
            Self {
                ix: perm.permute_ix(self.ix),
            }
        }

        #[inline]
        pub fn invert(&self, comb: Combination, permuted_com_cc: CenterOfMassCalcCache) -> Self {
            Self {
                ix: comb.invert_ix(self.ix, permuted_com_cc.dn),
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
            let mut bb = BoundingBox::new(added.add_ix().ix());
            bb.add_ix(Ix3(1, 1, 1));
            Self {
                ix: added.com().ix() - bb.min() * added.n().num(),
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct CenterOfMassCalcCache {
        dn: Ix3,
    }

    impl CenterOfMassCalcCache {
        pub fn new(d: Dimensions, n: N) -> Self {
            Self {
                dn: (d.ix() - Ix3(1, 1, 1)) * n.num(),
            }
        }
    }

    impl HasIx for CenterOfMass {
        fn ix(&self) -> Ix3 {
            self.ix
        }
    }

    impl PartialOrd for CenterOfMass {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CenterOfMass {
        fn cmp(&self, other: &Self) -> Ordering {
            self.ix.into_pattern().cmp(&other.ix.into_pattern())
        }
    }
}

mod perm {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct Permutation([usize; 3]);

    impl Permutation {
        pub fn all() -> &'static [Self; 6] {
            &[
                Self([0, 1, 2]),
                Self([1, 0, 2]),
                Self([2, 0, 1]),
                Self([0, 2, 1]),
                Self([1, 2, 0]),
                Self([2, 1, 0]),
            ]
        }

        #[inline]
        pub fn all_valid(inverted_com: CenterOfMass, dim: Dimensions) -> Vec<Self> {
            let d = dim.ix();
            let c = inverted_com.ix();
            let mut idc = [[0, d[0], c[0]], [1, d[1], c[1]], [2, d[2], c[2]]];

            // valid dimension takes precedence of canonical com
            idc.sort_unstable_by_key(|[_, d, c]| (*d, *c));

            let i = idc.map(|[i, ..]| i);
            let dc = idc.map(|[_, d, c]| [d, c]);

            Self::populate_vec_from_i_vals(i, dc)
        }

        #[inline]
        fn populate_vec_from_i_vals<T: Eq>(mut i: [usize; 3], v: [T; 3]) -> Vec<Self> {
            let mut out = Vec::with_capacity(6);
            out.push(Self(i));

            // indicies can be permuted while preserving order only if there are duplicates
            match (v[0] == v[1], v[1] == v[2]) {
                (false, false) => {}
                (true, false) => {
                    i.swap(0, 1);
                    out.push(Self(i));
                }
                (false, true) => {
                    i.swap(1, 2);
                    out.push(Self(i));
                }
                (true, true) => {
                    return Self::all().to_vec();
                }
            }

            out
        }

        pub fn indicies(&self) -> [usize; 3] {
            self.0
        }

        pub fn permute_ix(&self, ix: Ix3) -> Ix3 {
            let [x, y, z] = self.0;
            Ix3(ix[x], ix[y], ix[z])
        }

        #[inline]
        pub fn permute_view<'a>(&self, view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            view.permuted_axes(self.0)
        }
    }
}

mod comb {
    use super::*;

    // true represents inversion
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct Combination([bool; 3]);

    impl Combination {
        #[allow(dead_code)]
        pub fn all() -> &'static [Self; 8] {
            &[
                Self([false, false, false]),
                Self([true, false, false]),
                Self([false, true, false]),
                Self([false, false, true]),
                Self([true, true, false]),
                Self([true, false, true]),
                Self([false, true, true]),
                Self([true, true, true]),
            ]
        }

        #[inline]
        pub fn all_valid_octant(com: CenterOfMass, com_cc: CenterOfMassCalcCache) -> Vec<Self> {
            let ords = com.classify_octant(com_cc);

            Self::populate_vec_from_ord(ords)
        }

        #[inline]
        fn populate_vec_from_ord(ords: [Ordering; 3]) -> Vec<Self> {
            let mut out = Vec::with_capacity(8);
            out.push(ords.map(|o| o.is_gt()));

            for (i, o) in ords.into_iter().enumerate() {
                if o.is_eq() {
                    let r = 0..out.len();
                    out.extend_from_within(r.clone());
                    for b in &mut out[r] {
                        b[i] ^= true;
                    }
                }
            }

            out.into_iter().map(Self).collect()
        }

        pub fn mask(&self) -> [bool; 3] {
            self.0
        }

        pub fn invert_view<'a>(&self, mut view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            for (i, s) in self.0.into_iter().enumerate() {
                if s {
                    view.invert_axis(Axis(i))
                }
            }
            view
        }

        pub fn invert_ix(&self, ix: Ix3, dim: Ix3) -> Ix3 {
            let mut ix: [usize; 3] = ix.into_pattern().into();
            let dim: [usize; 3] = dim.into_pattern().into();

            for i in 0..3 {
                if self.0[i] {
                    ix[i] = dim[i] - ix[i];
                }
            }

            usize3_to_ix(ix)
        }
    }
}

mod n {
    use super::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash)]
    pub struct N(usize);

    impl N {
        // warning: O(n)
        pub fn from_view(view: ArrayView3<bool>) -> Self {
            Self(view.iter().filter(|&&b| b).count())
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

        pub fn new_min_max(min: Ix3, max: Ix3) -> Self {
            let (min, max) = (min.into_pattern().into(), max.into_pattern().into());
            debug_assert!(min <= max);
            Self { min, max }
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

        pub fn slice_view<'a>(&self, view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            let [xmin, ymin, zmin] = self.min;
            let [xmax, ymax, zmax] = self.max;
            view.slice_move(ndarray::s![xmin..=xmax, ymin..=ymax, zmin..=zmax])
        }

        pub fn min(&self) -> Ix3 {
            usize3_to_ix(self.min)
        }

        #[allow(dead_code)]
        pub fn max(&self) -> Ix3 {
            usize3_to_ix(self.max)
        }
    }
}

mod ap {
    use super::*;

    pub struct ArticulationPoints {
        ap_array: Array3<bool>,
    }

    impl ArticulationPoints {
        pub fn from_view(view: ArrayView3<bool>) -> Self {
            Self {
                ap_array: tarjans_algorithm(view, Self::find_root(view)),
            }
        }

        #[allow(dead_code)]
        pub fn can_remove(&self, ix: Ix3) -> bool {
            !self.ap_array[ix]
        }

        pub fn can_remove_one(view: ArrayView3<bool>, n: N, ix: Ix3) -> bool {
            search_connected_ignore_start(view, ix) == n.num()
        }

        fn find_root(view: ArrayView3<bool>) -> Ix3 {
            view.indexed_iter()
                .find_map(|(ix, b)| {
                    if *b {
                        Some(usize3_to_ix(ix.into()))
                    } else {
                        None
                    }
                })
                .unwrap()
        }

        pub fn ap_view(&self) -> ArrayView3<bool> {
            self.ap_array.view()
        }
    }

    // from https://en.wikipedia.org/wiki/Biconnected_component#Pseudocode
    fn tarjans_algorithm(view: ArrayView3<bool>, root: Ix3) -> Array3<bool> {
        #[derive(Clone)]
        struct State {
            depth: usize,
            lowest: usize,
            children: usize,
            is_articulation: bool,
            parent: Option<Ix3>,
        }

        // Some(State) means visited
        let mut states: Array3<Option<State>> = Array3::from_elem(view.raw_dim(), None);
        let mut out_ap: Array3<bool> = Array3::from_elem(view.raw_dim(), false);

        recurse(&view, &mut states, &mut out_ap, root, None, 0);

        fn recurse(
            view: &ArrayView3<bool>,
            states: &mut Array3<Option<State>>,
            out_ap: &mut Array3<bool>,
            ix: Ix3,
            parent: Option<Ix3>,
            depth: usize,
        ) {
            let before = states[ix].replace(State {
                depth,
                lowest: depth,
                children: 0,
                is_articulation: false,
                parent,
            });

            debug_assert!(before.is_none());

            for neighbor in neighbor_ixs(ix) {
                if !view.get(neighbor).unwrap_or(&false) {
                    continue;
                }

                if let Some(neighbor_state) = &states[neighbor] {
                    let ix_state = states[ix].as_ref().unwrap();

                    if ix_state.parent.map(|p| neighbor != p).unwrap_or(true) {
                        let neighbor_depth = neighbor_state.depth;

                        let ix_state = states[ix].as_mut().unwrap();
                        ix_state.lowest = ix_state.lowest.min(neighbor_depth);
                    }
                } else {
                    recurse(view, states, out_ap, neighbor, Some(ix), depth + 1);
                    let neighbor_lowest = states[neighbor].as_ref().unwrap().lowest;

                    let ix_state = states[ix].as_mut().unwrap();
                    ix_state.children += 1;
                    if neighbor_lowest >= ix_state.depth {
                        ix_state.is_articulation = true;
                    }
                    ix_state.lowest = ix_state.lowest.min(neighbor_lowest);
                }
            }

            let ix_state = states[ix].as_ref().unwrap();
            if (ix_state.parent.is_some() && ix_state.is_articulation)
                || (ix_state.parent.is_none() && ix_state.children > 1)
            {
                out_ap[ix] = true;
            }
        }

        out_ap
    }

    fn search_connected_ignore_start(view: ArrayView3<bool>, to_remove: Ix3) -> usize {
        thread_local! {
            static CACHE: RefCell<(Vec<bool>, Vec<Ix3>)> = RefCell::default();
        }

        CACHE.with(|rc| {
            let (seen, to_see) = &mut *rc.borrow_mut();
            seen.clear();
            seen.resize(view.len(), false);
            let mut seen =
                ndarray::ArrayViewMut3::from_shape(view.raw_dim(), seen.as_mut_slice()).unwrap();

            // pre visit removed node to prevent rediscovering it
            seen[to_remove] = true;

            to_see.clear();

            // start visiting on neighbor, if connected will discover a different connecting path to visit rest of nodes
            to_see.push(
                neighbor_ixs(to_remove)
                    .into_iter()
                    .find(|&ix| *view.get(ix).unwrap_or(&false))
                    .unwrap(),
            );

            while let Some(node) = to_see.pop() {
                seen[node] = true;

                for neighbor in neighbor_ixs(node) {
                    if view.get(neighbor).copied().unwrap_or(false) && !seen[neighbor] {
                        to_see.push(neighbor);
                    }
                }
            }

            seen.iter().filter(|&&b| b).count()
        })
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

#[allow(clippy::nonminimal_bool)]
fn is_neighbor(a: Ix3, b: Ix3) -> bool {
    let (ax, ay, az) = a.into_pattern();
    let (bx, by, bz) = b.into_pattern();

    let x = ax.abs_diff(bx) == 1;
    let y = ay.abs_diff(by) == 1;
    let z = az.abs_diff(bz) == 1;

    (x && !y && !z) || (!x && y && !z) || (!x && !y && z)
}

#[inline]
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

fn usize3_to_ix(array: [usize; 3]) -> Ix3 {
    let [x, y, z] = array;
    Ix3(x, y, z)
}

fn ix_cmp_elementwise(a: Ix3, b: Ix3) -> [Ordering; 3] {
    let (a1, a2, a3) = a.into_pattern();
    let (b1, b2, b3) = b.into_pattern();
    [a1.cmp(&b1), a2.cmp(&b2), a3.cmp(&b3)]
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

#[test]
fn see_ix_order() {
    let mut array = Array3::from_elem((2, 3, 4), Ix3(0, 0, 0));
    array.indexed_iter_mut().for_each(|(ix, val)| {
        *val = usize3_to_ix(ix.into());
    });
    println!("array: {:?}", array);
}
