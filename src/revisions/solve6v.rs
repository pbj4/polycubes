use {
    ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis, Dimension, Ix3},
    std::{cell::RefCell, cmp::Ordering, collections::BTreeMap, sync::atomic::AtomicUsize},
};

use {
    added::*, bb::*, com::*, comb::*, cropped::*, dim::*, n::*, padded::*, perm::*, transformed::*,
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
    reflection_symmetry: bool,
    out: &'a BTreeMap<usize, (AtomicUsize, AtomicUsize)>,
    target_n: usize,
    s: &rayon::Scope<'a>,
) {
    let current_n = transformed.n();

    let (r, p) = out.get(&current_n.num()).unwrap();
    if reflection_symmetry {
        p.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        r.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    if current_n.num() >= target_n {
        return;
    }

    let padded = Padded::from_cropped(&Cropped::from_transformed(*transformed));
    //let padded_aps = ArticulationPoints::from_view(padded.array_view());

    let streaming_aps =
        ap_streaming_2_wrapper::ApStreamingWrapper::from_view(padded.array_view(), padded.n());

    /*
    for (ix, _) in padded.array_view().indexed_iter() {
        let ix = usize3_to_ix(ix);
        debug_assert_eq!(padded_aps.can_remove(ix), streaming_aps.can_remove(ix));
    }
    */

    let mut added_backing_vec = Vec::new();

    let all_added = Added::from_padded_all(&padded, &mut added_backing_vec);

    let unique_addeds = {
        let dup = all_added.iter().map(|(added, add_ix)| {
            let cropped = Cropped::from_added(added);
            let add_ix = add_ix.ix() - cropped.bb_used().min();
            let (t, t_add_ix) = Transformed::from_cropped_min_point(&cropped, add_ix);
            (
                (t_add_ix.into_pattern(), t),
                cropped.bb_used(), //padded_aps.crop_transform(cropped.bb_used(), t.transform_used()),
            )
        });

        // this is faster than sort then dedup for smaller sizes
        let mut dedup = Vec::with_capacity(all_added.len());
        for elem in dup {
            if !dedup.iter().any(|(k, _)| k == &elem.0) {
                dedup.push(elem);
            }
        }
        dedup
    };

    'added: for ((add_ix, added_transformed), bb_used) in unique_addeds {
        /*
        let transformed_aps =
            padded_aps.crop_transform(bb_used, added_transformed.transform_used());
        */

        let add_ix = usize3_to_ix(add_ix);
        let added_transformed_view = added_transformed.array_view();

        let transformed_streaming_aps = {
            let mut transformed_streaming_aps = streaming_aps
                .clone_capacity(current_n.num() + 1)
                .crop_transform(bb_used, added_transformed.transform_used());
            transformed_streaming_aps.add_node(add_ix);
            transformed_streaming_aps
        };

        /*
        let addix_on_bridge = neighbor_ixs(add_ix)
            .into_iter()
            .filter(|&ix| *added_transformed_view.get(ix).unwrap_or(&false))
            .count()
            == 1
            && added_transformed.n().num() > 2;
        */

        let mut possible_smaller = added_transformed_view
            .indexed_iter()
            .filter_map(|(ix, b)| if *b { Some(usize3_to_ix(ix)) } else { None })
            .take_while(|ix| ix.into_pattern() < add_ix.into_pattern());

        /*
        let can_remove = possible_smaller.clone().any(|ix| {
            transformed_aps.can_remove(ix) && !(addix_on_bridge && is_neighbor(ix, add_ix))
        }) || possible_smaller.clone().any(|ix| {
            ArticulationPoints::can_remove_one(added_transformed_view, added_transformed.n(), ix)
        });
        */

        let can_remove_2 = possible_smaller.any(|ix| transformed_streaming_aps.can_remove(ix));

        //debug_assert_eq!(can_remove, can_remove_2);

        if can_remove_2 {
            continue 'added;
        }

        let transformed_owned = added_transformed.to_owned();
        s.spawn(move |s| {
            let transformed = Transformed::from_owned(&transformed_owned);

            recurse_hashless_min_point(
                &transformed,
                transformed.has_reflection_symmetry(),
                out,
                target_n,
                s,
            );
        })
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
        pub fn from_added(added: &Added<'a>) -> Self {
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

        // takes any vec and fills it with addixs
        #[allow(clippy::type_complexity)]
        pub fn populate_addixs(&self, out: &mut Vec<AddIx>) {
            thread_local! {
                static ALLOC: RefCell<(Vec<(bool, bool)>, Vec<Ix3>)> = RefCell::default();
            }

            ALLOC.with(|rc| {
                let (seen, to_see) = &mut *rc.borrow_mut();
                // (seen node, seen neighbor)
                seen.clear();
                seen.resize(self.array.len(), (false, false));
                let mut seen =
                    ArrayViewMut3::from_shape(self.array.raw_dim(), seen.as_mut_slice()).unwrap();

                to_see.clear();
                to_see.push(
                    self.array
                        .indexed_iter()
                        .find_map(|(ix, b)| if *b { Some(usize3_to_ix(ix)) } else { None })
                        .unwrap(),
                );

                while let Some(node) = to_see.pop() {
                    seen[node].0 = true;

                    for neighbor in neighbor_ixs(node) {
                        if let Some((seen, is_neighbor)) = seen.get_mut(neighbor) {
                            *is_neighbor = true;

                            if !*seen && self.array[neighbor] {
                                to_see.push(neighbor);
                            }
                        }
                    }
                }

                out.clear();
                out.reserve(4 * self.n.num() + 2);
                out.extend(
                    seen.indexed_iter()
                        .filter_map(|(ix, (is_node, is_neighbor))| {
                            if *is_neighbor && !is_node {
                                Some(AddIx {
                                    ix: usize3_to_ix(ix),
                                })
                            } else {
                                None
                            }
                        }),
                );
            });
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

    #[derive(Debug)]
    pub struct Added<'a> {
        view: ArrayView3<'a, bool>,
        com: CenterOfMass,
        add_ix: AddIx,
        n: N,
    }

    impl<'a> Added<'a> {
        pub fn from_padded_all(padded: &Padded, vec: &'a mut Vec<bool>) -> Vec<(Self, AddIx)> {
            thread_local! {
                static ALLOC: RefCell<Vec<AddIx>> = RefCell::default();
            }

            ALLOC.with(|rc| {
                let add_ixs = &mut *rc.borrow_mut();
                padded.populate_addixs(add_ixs);

                vec.clear();
                vec.reserve(padded.array_view().len() * (4 * padded.n().num() + 2));
                vec.extend_from_slice(padded.array_view().to_slice().unwrap());

                let original_len = vec.len();
                for _ in 0..add_ixs.len() - 1 {
                    vec.extend_from_within(0..original_len);
                }

                add_ixs
                    .iter()
                    .copied()
                    .zip(vec.chunks_exact_mut(original_len))
                    .map(|(add_ix, slice)| (Self::from_padded_slice(padded, add_ix, slice), add_ix))
                    .collect()
            })
        }

        // assumes slice to be preinitialized to padded in standard order
        fn from_padded_slice(padded: &Padded, add_ix: AddIx, slice: &'a mut [bool]) -> Self {
            let com = CenterOfMass::from_padded_to_added(padded, add_ix);
            let n = N::from_padded_to_added(padded, add_ix);

            let shape = padded.array_view().raw_dim();

            let mut view_mut = ArrayViewMut3::from_shape(shape, slice).unwrap();
            view_mut[add_ix.ix()] = true;

            let new = Self {
                view: ArrayView3::from_shape(shape, &*slice).unwrap(),
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

    impl<'a, 'b> HasView<'a, 'b> for Added<'a> {
        fn array_view(&self) -> ArrayView3<'a, bool> {
            self.view
        }
    }

    impl<'a, 'b> HasCom<'a, 'b> for Added<'a> {
        fn com(&self) -> CenterOfMass {
            self.com
        }
    }

    impl<'a, 'b> HasN<'a, 'b> for Added<'a> {
        fn n(&self) -> N {
            self.n
        }
    }
}

mod transformed {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct Transformed<'a> {
        com: CenterOfMass,
        view: ArrayView3<'a, bool>,
        n: N,

        // not checked for canonicity
        transform: Transformation,
    }

    pub struct TransformedOwned {
        n: N,
        com: CenterOfMass,
        array: Array3<bool>,
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

            thread_local! {
                static ALLOC: RefCell<(Vec<Combination>, Vec<Permutation>)> = RefCell::default();
            }

            ALLOC.with(|rc| {
                let (valid_combs, valid_perms) = &mut *rc.borrow_mut();

                Combination::populate_vec_all_valid_octant(
                    cropped.com(),
                    cropped.com_cc(),
                    valid_combs,
                );
                let valid_octal_com = cropped.com().invert(valid_combs[0], cropped.com_cc());

                Permutation::populate_vec_all_valid(valid_octal_com, cropped.dim(), valid_perms);
                let valid_com = valid_octal_com.permute(valid_perms[0]);

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

                for comb in valid_combs.iter().copied() {
                    for perm in valid_perms.iter().copied() {
                        let transform = Transformation { comb, perm };

                        let new = Self {
                            view: transform.transform_view(cropped.array_view()),
                            com: valid_com,
                            n,
                            transform,
                        };

                        let chirality = is_reflection(transform);

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
            })
        }

        pub fn from_cropped_min_point(cropped: &Cropped<'a>, ix: Ix3) -> (Self, Ix3) {
            let mut lowest: Option<((OrdView, [usize; 3]), Transformation)> = None;

            let n = cropped.n();

            thread_local! {
                static ALLOC: RefCell<(Vec<Combination>, Vec<Permutation>)> = RefCell::default();
            }

            ALLOC.with(|rc| {
                let (valid_combs, valid_perms) = &mut *rc.borrow_mut();

                Combination::populate_vec_all_valid_octant(
                    cropped.com(),
                    cropped.com_cc(),
                    valid_combs,
                );
                let valid_octal_com = cropped.com().invert(valid_combs[0], cropped.com_cc());

                Permutation::populate_vec_all_valid(valid_octal_com, cropped.dim(), valid_perms);
                let valid_com = valid_octal_com.permute(valid_perms[0]);

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

                for comb in valid_combs.iter().copied() {
                    for perm in valid_perms.iter().copied() {
                        let transform = Transformation { comb, perm };

                        let new_view = OrdView {
                            view: transform.transform_view(cropped.array_view()),
                        };

                        let transformed_ix = perm
                            .permute_ix(comb.invert_ix(ix, cropped.dim().ix() - Ix3(1, 1, 1)))
                            .into_pattern()
                            .into();

                        if let Some((prev_lowest, _prev_transform)) = lowest {
                            let new_lowest = (new_view, transformed_ix);

                            if new_lowest < prev_lowest {
                                lowest = Some((new_lowest, transform))
                            }
                        } else {
                            lowest = Some(((new_view, transformed_ix), transform));
                        }
                    }
                }

                let ((OrdView { view }, transformed_ix), transform) = lowest.unwrap();
                let new = Self {
                    view,
                    n,
                    com: valid_com,
                    transform,
                };
                new.verify_com();
                new.verify_n();
                (new, usize3_to_ix(transformed_ix.into()))
            })
        }

        pub fn has_reflection_symmetry(&self) -> bool {
            thread_local! {
                static ALLOC: RefCell<(Vec<Combination>, Vec<Permutation>)> = RefCell::default();
            }

            ALLOC.with(|rc| {
                let (valid_combs, valid_perms) = &mut *rc.borrow_mut();

                let com_cc = CenterOfMassCalcCache::new(self.dim(), self.n());

                Combination::populate_vec_all_valid_octant(self.com(), com_cc, valid_combs);
                let valid_octal_com = self.com().invert(valid_combs[0], com_cc);

                Permutation::populate_vec_all_valid(valid_octal_com, self.dim(), valid_perms);

                // fast path
                if valid_combs.len() == 1 && valid_perms.len() == 1 {
                    return false;
                }

                for comb in valid_combs.iter().copied() {
                    for perm in valid_perms.iter().copied() {
                        let transform = Transformation { comb, perm };

                        if is_reflection(transform)
                            && transform.transform_view(self.view) == self.view
                        {
                            return true;
                        }
                    }
                }

                false
            })
        }

        pub fn transform_used(&self) -> Transformation {
            self.transform
        }

        pub fn to_owned(self) -> TransformedOwned {
            TransformedOwned {
                n: self.n,
                com: self.com,
                array: self.view.to_owned(),
                transform: self.transform,
            }
        }

        pub fn from_owned(owned: &'a TransformedOwned) -> Self {
            Self {
                n: owned.n,
                com: owned.com,
                view: owned.array.view(),
                transform: owned.transform,
            }
        }
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
    struct OrdView<'a> {
        view: ArrayView3<'a, bool>,
    }

    impl<'a> PartialOrd for OrdView<'a> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a> Ord for OrdView<'a> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.view.iter().cmp(other.view.iter())
        }
    }

    impl Transformation {
        #[inline]
        pub fn transform_view<'a>(&self, view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            self.perm.permute_view(self.comb.invert_view(view))
        }

        pub fn transform_array<T>(&self, array: Array3<T>) -> Array3<T> {
            self.perm.permute_array(self.comb.invert_array(array))
        }

        pub fn comb(&self) -> Combination {
            self.comb
        }

        pub fn perm(&self) -> Permutation {
            self.perm
        }
    }

    impl<'a> PartialOrd for Transformed<'a> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a> Ord for Transformed<'a> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.com
                .cmp(&other.com)
                .then_with(|| self.view.iter().cmp(other.view.iter()))
        }
    }

    impl<'a, 'b> PartialEq<Transformed<'b>> for Transformed<'a> {
        fn eq(&self, other: &Transformed<'b>) -> bool {
            self.com.eq(&other.com) && self.view.eq(&other.view)
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
                    .filter_map(|(ix, &b)| b.then_some(usize3_to_ix(ix)))
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

        #[allow(dead_code)]
        pub fn octant_is_valid(&self, com_cc: CenterOfMassCalcCache) -> bool {
            let [x, y, z] = self.classify_octant(com_cc);
            x.is_le() && y.is_le() && z.is_le()
        }

        pub fn classify_octant(&self, com_cc: CenterOfMassCalcCache) -> [Ordering; 3] {
            ix_cmp_elementwise(self.ix * 2, com_cc.dn)
        }

        #[allow(dead_code)]
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

        // takes any vec and fills it with valid permutations
        #[inline]
        pub fn populate_vec_all_valid(
            inverted_com: CenterOfMass,
            dim: Dimensions,
            out: &mut Vec<Self>,
        ) {
            let d = dim.ix();
            let c = inverted_com.ix();
            let mut idc = [[0, d[0], c[0]], [1, d[1], c[1]], [2, d[2], c[2]]];

            // valid dimension takes precedence of canonical com
            idc.sort_by_key(|[_, d, c]| (*d, *c));

            let i = idc.map(|[i, ..]| i);
            let dc = idc.map(|[_, d, c]| [d, c]);

            Self::populate_vec_from_i_vals(i, dc, out);
        }

        #[inline]
        fn populate_vec_from_i_vals<T: Eq>(mut i: [usize; 3], v: [T; 3], out: &mut Vec<Self>) {
            out.clear();
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
                    *out = Self::all().to_vec();
                }
            }
        }

        pub fn indicies(&self) -> [usize; 3] {
            self.0
        }

        #[inline]
        pub fn permute_ix(&self, ix: Ix3) -> Ix3 {
            let [x, y, z] = self.0;
            Ix3(ix[x], ix[y], ix[z])
        }

        #[inline]
        pub fn permute_view<'a>(&self, view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            view.permuted_axes(self.0)
        }

        pub fn permute_array<T>(&self, array: Array3<T>) -> Array3<T> {
            array.permuted_axes(self.0)
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

        // takes any vec and fills it with valid octant combinations
        #[inline]
        pub fn populate_vec_all_valid_octant(
            com: CenterOfMass,
            com_cc: CenterOfMassCalcCache,
            out: &mut Vec<Self>,
        ) {
            let ords = com.classify_octant(com_cc);

            Self::populate_vec_from_ord(ords, out);
        }

        #[inline]
        fn populate_vec_from_ord(ords: [Ordering; 3], out: &mut Vec<Self>) {
            out.clear();
            out.push(Self(ords.map(|o| o.is_gt())));

            for (i, o) in ords.into_iter().enumerate() {
                if o.is_eq() {
                    let r = 0..out.len();
                    out.extend_from_within(r.clone());
                    for b in &mut out[r] {
                        b.0[i] ^= true;
                    }
                }
            }
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

        pub fn invert_array<T>(&self, mut array: Array3<T>) -> Array3<T> {
            for (i, s) in self.0.into_iter().enumerate() {
                if s {
                    array.invert_axis(Axis(i))
                }
            }
            array
        }

        #[inline]
        pub fn invert_ix(&self, ix: Ix3, dim: Ix3) -> Ix3 {
            let mut ix: [usize; 3] = ix.into_pattern().into();
            let dim: [usize; 3] = dim.into_pattern().into();

            for i in 0..3 {
                if self.0[i] {
                    ix[i] = dim[i] - ix[i];
                }
            }

            usize3_to_ix(ix.into())
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

        #[inline]
        pub fn slice_view<'a>(&self, mut view: ArrayView3<'a, bool>) -> ArrayView3<'a, bool> {
            let [xmin, ymin, zmin] = self.min;
            let [xmax, ymax, zmax] = self.max;
            view.slice_axis_inplace(Axis(0), (xmin..=xmax).into());
            view.slice_axis_inplace(Axis(1), (ymin..=ymax).into());
            view.slice_axis_inplace(Axis(2), (zmin..=zmax).into());
            view
        }

        pub fn slice_array<T>(&self, mut array: Array3<T>) -> Array3<T> {
            let [xmin, ymin, zmin] = self.min;
            let [xmax, ymax, zmax] = self.max;
            array.slice_axis_inplace(Axis(0), (xmin..=xmax).into());
            array.slice_axis_inplace(Axis(1), (ymin..=ymax).into());
            array.slice_axis_inplace(Axis(2), (zmin..=zmax).into());
            array
        }

        pub fn min(&self) -> Ix3 {
            usize3_to_ix(self.min.into())
        }

        #[allow(dead_code)]
        pub fn max(&self) -> Ix3 {
            usize3_to_ix(self.max.into())
        }
    }
}

#[allow(dead_code)]
mod ap {
    use super::*;

    pub struct ArticulationPoints {
        ap_array: Array3<bool>,
    }

    pub struct ArticulationPointsView<'a> {
        ap_view: ArrayView3<'a, bool>,
    }

    impl ArticulationPoints {
        pub fn from_view(view: ArrayView3<bool>) -> Self {
            Self {
                ap_array: tarjans_algorithm(view, Self::find_root(view)),
            }
        }

        pub fn crop_transform(&self, bb: BoundingBox, t: Transformation) -> ArticulationPointsView {
            ArticulationPointsView {
                ap_view: t.transform_view(bb.slice_view(self.ap_array.view())),
            }
        }

        pub fn can_remove_one(view: ArrayView3<bool>, n: N, ix: Ix3) -> bool {
            search_connected_ignore_start(view, ix) == n.num()
        }

        pub fn can_remove(&self, ix: Ix3) -> bool {
            !self.ap_array[ix]
        }

        fn find_root(view: ArrayView3<bool>) -> Ix3 {
            view.indexed_iter()
                .find_map(|(ix, b)| if *b { Some(usize3_to_ix(ix)) } else { None })
                .unwrap()
        }
    }

    impl<'a> ArticulationPointsView<'a> {
        pub fn can_remove(&self, ix: Ix3) -> bool {
            !self.ap_view[ix]
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

        fn recurse(
            view: &ArrayView3<bool>,
            states: &mut ArrayViewMut3<Option<State>>,
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

        thread_local! {
            static ALLOC: RefCell<Vec<Option<State>>> = RefCell::default();
        }

        ALLOC.with(|rc| {
            let states = &mut *rc.borrow_mut();

            states.fill(None);
            states.resize(view.len(), None);

            let mut states =
                ndarray::ArrayViewMut::from_shape(view.raw_dim(), states.as_mut_slice()).unwrap();
            let mut out_ap: Array3<bool> = Array3::from_elem(view.raw_dim(), false);

            recurse(&view, &mut states, &mut out_ap, root, None, 0);

            out_ap
        })
    }

    fn search_connected_ignore_start(view: ArrayView3<bool>, to_remove: Ix3) -> usize {
        thread_local! {
            static ALLOC: RefCell<(Vec<bool>, Vec<Ix3>)> = RefCell::default();
        }

        ALLOC.with(|rc| {
            let (seen, to_see) = &mut *rc.borrow_mut();
            seen.clear();
            seen.resize(view.len(), false);
            let mut seen = ArrayViewMut3::from_shape(view.raw_dim(), seen.as_mut_slice()).unwrap();

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

mod ap_semi_streaming {
    use super::*;

    trait UnionFind<T: PartialEq> {
        fn data_push(&mut self, data: (T, usize));
        fn data_slice(&self) -> &[(T, usize)];
        fn data_slice_mut(&mut self) -> &mut [(T, usize)];

        fn find(&self, data: &T) -> Option<&(T, usize)> {
            self.data_slice().iter().find(|(d, _)| d == data)
        }

        fn make_set(&mut self, data: T) {
            debug_assert!(self.find(&data).is_none());
            let id = self.data_slice().len();
            self.data_push((data, id));
        }

        fn same_set(&self, a: &T, b: &T) -> bool {
            self.find(a).unwrap().1 == self.find(b).unwrap().1
        }

        fn union(&mut self, a: &T, b: &T) {
            let a = self.find(a).unwrap().1;
            let b = self.find(b).unwrap().1;

            for (_, rep) in self.data_slice_mut() {
                if *rep == a {
                    *rep = b;
                }
            }
        }

        fn all_one_set(&self) -> bool {
            if let Some((_, id)) = self.data_slice().first() {
                self.data_slice().iter().all(|(_, i)| i == id)
            } else {
                true
            }
        }
    }

    #[derive(Debug, Clone)]
    struct StaticUnionFind<T, const C: usize> {
        array: heapless::Vec<(T, usize), C>,
    }

    impl<T: PartialEq, const C: usize> StaticUnionFind<T, C> {
        pub fn new() -> Self {
            Self {
                array: heapless::Vec::new(),
            }
        }
    }

    impl<T: PartialEq + std::fmt::Debug, const C: usize> UnionFind<T> for StaticUnionFind<T, C> {
        fn data_slice(&self) -> &[(T, usize)] {
            self.array.as_slice()
        }

        fn data_slice_mut(&mut self) -> &mut [(T, usize)] {
            self.array.as_mut()
        }

        fn data_push(&mut self, data: (T, usize)) {
            self.array.push(data).unwrap();
        }
    }

    #[derive(Debug, Clone)]
    struct HeapUnionFind<T> {
        vec: Vec<(T, usize)>, // (data, rep)
    }

    impl<T: PartialEq + Clone> HeapUnionFind<T> {
        pub fn new(capacity: usize) -> Self {
            Self {
                vec: Vec::with_capacity(capacity),
            }
        }

        pub fn clone_capacity(&self, capacity: usize) -> Self {
            let mut vec = Vec::with_capacity(capacity);
            vec.extend_from_slice(self.vec.as_slice());
            Self { vec }
        }
    }

    impl<T: PartialEq> UnionFind<T> for HeapUnionFind<T> {
        fn data_slice(&self) -> &[(T, usize)] {
            self.vec.as_slice()
        }

        fn data_slice_mut(&mut self) -> &mut [(T, usize)] {
            self.vec.as_mut_slice()
        }

        fn data_push(&mut self, data: (T, usize)) {
            self.vec.push(data)
        }
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct NodeId(usize);

    #[derive(Debug, Clone)]
    struct NodeData {
        parent: NodeId,
        neighbors: StaticUnionFind<NodeId, 6>,
    }

    #[derive(Clone)]
    pub struct ApStreaming {
        cc: HeapUnionFind<NodeId>,
        nodes: Vec<NodeData>,
    }

    impl ApStreaming {
        pub fn new(capacity: usize) -> Self {
            Self {
                cc: HeapUnionFind::new(capacity),
                nodes: Vec::with_capacity(capacity),
            }
        }

        pub fn clone_capacity(&self, capacity: usize) -> Self {
            let mut nodes = Vec::with_capacity(capacity);
            nodes.extend_from_slice(self.nodes.as_slice());
            Self {
                cc: HeapUnionFind::clone_capacity(&self.cc, capacity),
                nodes,
            }
        }

        pub fn add_node(&mut self) -> NodeId {
            let id = NodeId(self.nodes.len());
            self.cc.make_set(id);
            self.nodes.push(NodeData {
                parent: id,
                neighbors: StaticUnionFind::new(),
            });
            id
        }

        /*
        pub fn add_edge(&mut self, a: NodeId, b: NodeId) {
            if self.cc.same_set(&a, &b) {
                // slow version
                let mut a_path = self.path_to_root(a);
                let mut b_path = self.path_to_root(b);

                let mut lca = *a_path.last().unwrap();
                let b_last = *b_path.last().unwrap();
                debug_assert_eq!(lca, b_last, "a: {a:?}, b: {b:?}");

                a_path.pop();
                b_path.pop();

                while a_path.last() == b_path.last() {
                    lca = a_path.pop().unwrap();
                    let b_last = b_path.pop().unwrap();
                    debug_assert_eq!(lca, b_last);
                }

                a_path.push(lca);
                b_path.reverse();
                a_path.append(&mut b_path);

                for triple in a_path.windows(3) {
                    let [i, j, k]: [NodeId; 3] = triple.try_into().unwrap();
                    self.nodes[j.0].neighbors.union(&i, &k);
                }
            } else {
                self.cc.union(&a, &b);

                // equivalant to dynamic streaming spanning tree???
                if self.nodes[b.0].parent == b {
                    self.nodes[b.0].parent = a;
                } else if self.nodes[a.0].parent == a {
                    self.nodes[a.0].parent = b;
                } else {
                    unreachable!(
                        "a: {:?}, a.parent: {:?}, b: {:?}, b.parent: {:?}, self.nodes: {:#?}",
                        a, self.nodes[a.0].parent, b, self.nodes[b.0].parent, self.nodes
                    );
                }

                self.nodes[a.0].neighbors.make_set(b);
                self.nodes[b.0].neighbors.make_set(a);
            }
        }
        */

        // a is preexisting, b is new
        pub fn add_edge(&mut self, a: NodeId, b: NodeId) {
            if self.cc.same_set(&a, &b) {
                thread_local! {
                    static ALLOC: RefCell<(Vec<NodeId>, Vec<NodeId>)> = RefCell::default();
                }

                ALLOC.with(|rc| {
                    let (a_path, b_path) = &mut *rc.borrow_mut();

                    self.populate_vec_path_to_root(a, a_path);
                    self.populate_vec_path_to_root(b, b_path);

                    let mut lca = *a_path.last().unwrap();
                    let b_last = *b_path.last().unwrap();
                    debug_assert_eq!(lca, b_last, "a: {a:?}, b: {b:?}");

                    a_path.pop();
                    b_path.pop();

                    while a_path.last() == b_path.last() {
                        lca = a_path.pop().unwrap();
                        let b_last = b_path.pop().unwrap();
                        debug_assert_eq!(lca, b_last);
                    }

                    a_path.push(lca);
                    b_path.reverse();
                    a_path.append(b_path);

                    for triple in a_path.windows(3) {
                        let [i, j, k]: [NodeId; 3] = triple.try_into().unwrap();
                        self.nodes[j.0].neighbors.union(&i, &k);
                    }
                });
            } else {
                self.cc.union(&a, &b);

                debug_assert_eq!(self.nodes[b.0].parent, b);
                self.nodes[b.0].parent = a;

                self.nodes[a.0].neighbors.make_set(b);
                self.nodes[b.0].neighbors.make_set(a);
            }
        }

        pub fn is_ap(&self, id: NodeId) -> bool {
            !self.nodes[id.0].neighbors.all_one_set()
        }

        /*
        fn path_to_root(&self, id: NodeId) -> Vec<NodeId> {
            let mut path = vec![id];

            while self.nodes[path.last().unwrap().0].parent != *path.last().unwrap() {
                path.push(self.nodes[path.last().unwrap().0].parent);
            }

            path
        }
        */

        fn populate_vec_path_to_root(&self, id: NodeId, path: &mut Vec<NodeId>) {
            path.clear();
            path.push(id);

            while self.nodes[path.last().unwrap().0].parent != *path.last().unwrap() {
                path.push(self.nodes[path.last().unwrap().0].parent);
            }
        }
    }
}

mod ap_streaming_2_wrapper {
    use super::{ap_semi_streaming::*, *};

    #[derive(Clone)]
    pub struct ApStreamingWrapper {
        inner: ApStreaming,
        map: Array3<Option<NodeId>>,
    }

    impl ApStreamingWrapper {
        pub fn from_view(view: ArrayView3<bool>, n: N) -> Self {
            let mut inner = ApStreaming::new(n.num());
            let mut map = Array3::from_elem(view.raw_dim(), None);

            let root = view
                .indexed_iter()
                .find_map(|(ix, b)| if *b { Some(usize3_to_ix(ix)) } else { None })
                .unwrap();

            let mut to_see: Vec<(Ix3, Option<Ix3>)> = Vec::with_capacity(n.num());
            to_see.push((root, None));
            map[root] = Some(inner.add_node());

            while let Some((node, parent)) = to_see.pop() {
                let node_id = map[node].unwrap();

                for neighbor in neighbor_ixs(node) {
                    if *view.get(neighbor).unwrap_or(&false) && Some(neighbor) != parent {
                        let neighbor_id = *map[neighbor].get_or_insert_with(|| {
                            to_see.push((neighbor, Some(node)));
                            inner.add_node()
                        });
                        inner.add_edge(node_id, neighbor_id);
                    }
                }
            }

            Self { inner, map }
        }

        pub fn add_node(&mut self, add_ix: Ix3) {
            let new_id = self.inner.add_node();
            self.map[add_ix] = Some(new_id);

            for neighbor in neighbor_ixs(add_ix) {
                if let Some(neighbor_id) = self.map.get(neighbor).copied().flatten() {
                    self.inner.add_edge(neighbor_id, new_id)
                }
            }
        }

        pub fn crop_transform(self, bb: BoundingBox, transformation: Transformation) -> Self {
            Self {
                map: transformation.transform_array(bb.slice_array(self.map)),
                ..self
            }
        }

        pub fn clone_capacity(&self, capacity: usize) -> Self {
            Self {
                inner: self.inner.clone_capacity(capacity),
                map: self.map.clone(),
            }
        }

        pub fn can_remove(&self, ix: Ix3) -> bool {
            if let Some(id) = self.map[ix] {
                !self.inner.is_ap(id)
            } else {
                true
            }
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

#[allow(clippy::nonminimal_bool, dead_code)]
#[inline]
fn is_neighbor(a: Ix3, b: Ix3) -> bool {
    let (ax, ay, az) = a.into_pattern();
    let (bx, by, bz) = b.into_pattern();

    let x = ax.abs_diff(bx) == 1;
    let y = ay.abs_diff(by) == 1;
    let z = az.abs_diff(bz) == 1;

    (x && !y && !z) || (!x && y && !z) || (!x && !y && z)
}

#[inline]
fn is_reflection(transformation: Transformation) -> bool {
    let comb = transformation.comb();
    let perm = transformation.perm();

    let comb = comb.mask();

    // shift x to first position (valid rotation)
    let mut perm = perm.indicies();
    let x_pos = perm.iter().position(|&p| p == 0).unwrap();
    perm.rotate_left(x_pos);

    // check for odd number of reflections
    // y and z being swapped and axis inversions are reflections
    (perm[1] != 1) ^ comb[0] ^ comb[1] ^ comb[2]
}

#[inline]
fn usize3_to_ix(tuple: (usize, usize, usize)) -> Ix3 {
    let (x, y, z) = tuple;
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
