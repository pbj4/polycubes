#![allow(dead_code)]

use {
    ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis, Dimension, Ix3},
    std::{cell::RefCell, cmp::Ordering, collections::BTreeMap},
};

use {
    added::*, ap_streaming_2_wrapper::*, bb::*, com::*, comb::*, cropped::*, dim::*, n::*,
    padded::*, perm::*, transformed::*,
};

#[cfg(not(feature = "output"))]
pub fn solve(n: usize) -> BTreeMap<usize, (usize, usize)> {
    let base = Transformed::default();
    let counts = Counts::new(n);

    recurse_hashless_min_point(base, n, &counts);

    println!("results:");

    let counts = counts.into_map();

    for (k, (r, p)) in counts.iter() {
        println!("n: {:?}, r: {:?}, p: {:?}", k, r, p);
    }

    counts
}

#[cfg(feature = "output")]
pub fn solve_out(n: usize, out: &(impl Fn(ArrayView3<bool>) + Send + Sync)) {
    let base = Transformed::default();
    recurse_hashless_min_point(base, n, &out);
}

#[cfg(not(feature = "output"))]
pub fn solve_partial(n: usize, base: ArrayView3<bool>) -> BTreeMap<usize, (usize, usize)> {
    let base = Transformed::from_view_unchecked(base);
    let counts = Counts::new(n);

    recurse_hashless_min_point(base, n, &counts);

    counts.into_map()
}

fn recurse_hashless_min_point(
    transformed: Transformed,
    target_n: usize,
    #[cfg(not(feature = "output"))] counts: &Counts,
    #[cfg(feature = "output")] out: &(impl Fn(ArrayView3<bool>) + Send + Sync),
) {
    let current_n = transformed.n();

    #[cfg(not(feature = "output"))]
    counts.insert(current_n, transformed.has_reflection_symmetry());

    if current_n.num() == target_n {
        #[cfg(feature = "output")]
        out(transformed.array_view());
        return;
    }

    let padded = Padded::from_cropped(&Cropped::from_transformed(transformed));

    let mut streaming_aps = ApStreamingWrapper::from_view(padded.array_view(), padded.n());
    let streaming_aps_next_id = streaming_aps.add_node_unconnected();

    let mut added_backing_vec = Vec::new();
    let all_added = Added::from_padded_all(&padded, &mut added_backing_vec);

    let unique_added = collect_dedup(all_added.iter().map(|(added, add_ix)| {
        let cropped = Cropped::from_added(added);
        let c_add_ix = add_ix.ix() - cropped.bb_used().min();
        let (t, t_add_ix) = Transformed::from_cropped_min_point(&cropped, c_add_ix); // canonicalize added polycube
        ((t_add_ix.into_pattern(), t), (cropped.bb_used(), add_ix))
    }));

    rayon::scope(|s| {
        for ((t_add_ix, added_transformed), (bb_used, original_add_ix)) in unique_added {
            let t_add_ix = usize3_to_ix(t_add_ix);
            let added_transformed_view = added_transformed.array_view();

            // check for bridge
            let only_neighbor = {
                let mut iter = neighbor_ixs(t_add_ix)
                    .into_iter()
                    .filter(|&ix| *added_transformed_view.get(ix).unwrap_or(&false));

                iter.next()
                    .xor(iter.next()) // extract one if there's only one neighbor
                    .filter(|_| added_transformed.n().num() > 2) // n <= 2 doesn't count
            };

            let mut possible_smaller = added_transformed_view
                .indexed_iter()
                .filter_map(|(ix, b)| if *b { Some(usize3_to_ix(ix)) } else { None })
                .take_while(|ix| ix.into_pattern() < t_add_ix.into_pattern());

            let transformed_streaming_aps =
                streaming_aps.crop_transform_view(bb_used, added_transformed.transform_used());

            // fast path: check previous graph for non articulation points
            let can_remove_1 = possible_smaller
                .clone()
                .any(|ix| transformed_streaming_aps.can_remove(ix) && Some(ix) != only_neighbor);

            if can_remove_1 {
                continue;
            }

            // slow path: update streaming graph
            let can_remove_2 = {
                let mut streaming_aps = streaming_aps.clone_borrow();
                streaming_aps.connect_node(streaming_aps_next_id, *original_add_ix);
                streaming_aps.replace_view(transformed_streaming_aps);

                possible_smaller.any(|ix| streaming_aps.can_remove(ix))
            };

            if can_remove_2 {
                continue;
            }

            s.spawn(move |_| {
                #[cfg(not(feature = "output"))]
                recurse_hashless_min_point(added_transformed, target_n, counts);

                #[cfg(feature = "output")]
                recurse_hashless_min_point(added_transformed, target_n, out);
            })
        }
    });
}

// faster for smaller sizes
#[inline]
fn collect_dedup<K: Eq, V>(iter: impl ExactSizeIterator<Item = (K, V)>) -> Vec<(K, V)> {
    let mut dedup = Vec::with_capacity(iter.len());
    for elem in iter {
        if !dedup.iter().any(|(k, _)| k == &elem.0) {
            dedup.push(elem);
        }
    }
    dedup
}

struct Counts {
    counts: Vec<(
        std::sync::atomic::AtomicUsize,
        std::sync::atomic::AtomicUsize,
    )>,
}

impl Counts {
    fn new(n: usize) -> Self {
        Self {
            counts: std::iter::repeat_with(Default::default).take(n).collect(),
        }
    }

    fn insert(&self, n: N, reflection_symmetry: bool) {
        let (r, p) = &self.counts[n.num() - 1];
        if reflection_symmetry { p } else { r }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn into_map(self) -> BTreeMap<usize, (usize, usize)> {
        self.counts
            .into_iter()
            .enumerate()
            .map(|(n, (r, p))| {
                let (r, p) = (r.into_inner(), p.into_inner());
                (n + 1, (p + 2 * r, p + r))
            })
            .collect()
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
            bb.insert_ix(Ix3(1, 1, 1));
            bb.insert_ix(added.dim().ix() - Ix3(2, 2, 2));

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

    // 1x1x1 starting polycube
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

        // assumes slice to be preinitialized to padded polycube in standard memory order
        #[inline]
        fn from_padded_slice(padded: &Padded, add_ix: AddIx, slice: &'a mut [bool]) -> Self {
            let com = CenterOfMass::from_padded_to_added(padded, add_ix);
            let n = N::from_padded_to_added(padded, add_ix);

            let shape = padded.array_view().raw_dim();

            //let mut view_mut = ArrayViewMut3::from_shape(shape, slice).unwrap();
            // safety: above line works
            let mut view_mut = unsafe { ArrayViewMut3::from_shape_ptr(shape, slice.as_mut_ptr()) };
            view_mut[add_ix.ix()] = true;

            let new = Self {
                //view: ArrayView3::from_shape(shape, &*slice).unwrap(),
                // safety: above line works
                view: unsafe { ArrayView3::from_shape_ptr(shape, slice.as_ptr()) },
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

    // does inversion first, then permutation
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Transformation {
        comb: Combination,
        perm: Permutation,
    }

    impl<'a> Transformed<'a> {
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

        pub fn from_view_unchecked(view: ArrayView3<'a, bool>) -> Self {
            Self {
                com: CenterOfMass::from_view(view),
                n: N::from_view(view),
                view,
                transform: Transformation::default(),
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
        pub fn transform_view<'a, T>(&self, view: ArrayView3<'a, T>) -> ArrayView3<'a, T> {
            self.perm.permute_view(self.comb.invert_view(view))
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

    impl<'a> Default for Transformed<'a> {
        fn default() -> Self {
            let cropped = Cropped::default();
            Self {
                com: cropped.com(),
                view: cropped.array_view(),
                n: cropped.n(),
                transform: Transformation::default(),
            }
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

        pub fn classify_octant(&self, com_cc: CenterOfMassCalcCache) -> [Ordering; 3] {
            ix_cmp_elementwise(self.ix * 2, com_cc.dn)
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
            bb.insert_ix(Ix3(1, 1, 1));
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
        pub fn populate_vec_all_valid(
            inverted_com: CenterOfMass,
            dim: Dimensions,
            out: &mut Vec<Self>,
        ) {
            let d = dim.ix();
            let c = inverted_com.ix();
            let mut idc = [[0, d[0], c[0]], [1, d[1], c[1]], [2, d[2], c[2]]];

            // valid dimension takes precedence over canonical com
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
        pub fn permute_view<'a, T>(&self, view: ArrayView3<'a, T>) -> ArrayView3<'a, T> {
            view.permuted_axes(self.0)
        }
    }

    impl Default for Permutation {
        fn default() -> Self {
            Self([0, 1, 2])
        }
    }
}

mod comb {
    use super::*;

    // true represents inversion
    #[derive(Clone, Copy, PartialEq, Debug, Default)]
    pub struct Combination([bool; 3]);

    impl Combination {
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

        pub fn invert_view<'a, T>(&self, mut view: ArrayView3<'a, T>) -> ArrayView3<'a, T> {
            for (i, s) in self.0.into_iter().enumerate() {
                if s {
                    view.invert_axis(Axis(i))
                }
            }
            view
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

        pub fn insert_ix(&mut self, ix: Ix3) {
            let ix: [_; 3] = ix.into_pattern().into();
            for (i, ix) in ix.into_iter().enumerate() {
                self.min[i] = self.min[i].min(ix);
                self.max[i] = self.max[i].max(ix);
            }
        }

        #[inline]
        pub fn slice_view<'a, T>(&self, mut view: ArrayView3<'a, T>) -> ArrayView3<'a, T> {
            let [xmin, ymin, zmin] = self.min;
            let [xmax, ymax, zmax] = self.max;
            view.slice_axis_inplace(Axis(0), (xmin..=xmax).into());
            view.slice_axis_inplace(Axis(1), (ymin..=ymax).into());
            view.slice_axis_inplace(Axis(2), (zmin..=zmax).into());
            view
        }

        pub fn min(&self) -> Ix3 {
            usize3_to_ix(self.min.into())
        }
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

    #[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
    pub struct NodeId(usize);

    #[derive(Debug, Clone)]
    pub struct NodeData {
        parent: NodeId,
        neighbors: StaticUnionFind<NodeId, 6>,
    }

    #[derive(Clone)]
    pub struct ApStreaming {
        nodes: Vec<NodeData>,
    }

    pub trait ApStreamingInPlace {
        fn node_datas(&self) -> &[NodeData];
        fn node_datas_mut(&mut self) -> &mut [NodeData];

        // a is preexisting, b is new
        // relies on connected nodes to be added in consecutive order
        fn add_edge(&mut self, a: NodeId, b: NodeId) {
            if self.node_datas_mut()[b.0].parent != b {
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
                        self.node_datas_mut()[j.0].neighbors.union(&i, &k);
                    }
                });
            } else {
                self.node_datas_mut()[b.0].parent = a;

                self.node_datas_mut()[a.0].neighbors.make_set(b);
                self.node_datas_mut()[b.0].neighbors.make_set(a);
            }
        }

        fn populate_vec_path_to_root(&self, id: NodeId, path: &mut Vec<NodeId>) {
            path.clear();
            path.push(id);

            while self.node_datas()[path.last().unwrap().0].parent != *path.last().unwrap() {
                path.push(self.node_datas()[path.last().unwrap().0].parent);
            }
        }

        fn is_ap(&self, id: NodeId) -> bool {
            !self.node_datas()[id.0].neighbors.all_one_set()
        }
    }

    impl ApStreamingInPlace for ApStreaming {
        fn node_datas(&self) -> &[NodeData] {
            self.nodes.as_slice()
        }

        fn node_datas_mut(&mut self) -> &mut [NodeData] {
            self.nodes.as_mut_slice()
        }
    }

    impl ApStreaming {
        pub fn new(capacity: usize) -> Self {
            Self {
                nodes: Vec::with_capacity(capacity),
            }
        }

        pub fn add_node(&mut self) -> NodeId {
            let id = NodeId(self.nodes.len());
            self.nodes.push(NodeData {
                parent: id,
                neighbors: StaticUnionFind::new(),
            });
            id
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

    pub struct ApStreamingWrapperView<'a> {
        inner: &'a ApStreaming,
        map_view: ArrayView3<'a, Option<NodeId>>,
    }

    pub struct ApStreamingWrapperPartialView<'a> {
        inner: ApStreaming,
        map_view: ArrayView3<'a, Option<NodeId>>,
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

        pub fn add_node_unconnected(&mut self) -> NodeId {
            self.inner.add_node()
        }

        #[inline]
        pub fn crop_transform_view(
            &self,
            bb: BoundingBox,
            transformation: Transformation,
        ) -> ApStreamingWrapperView {
            ApStreamingWrapperView {
                inner: &self.inner,
                map_view: transformation.transform_view(bb.slice_view(self.map.view())),
            }
        }

        pub fn clone_borrow(&self) -> ApStreamingWrapperPartialView {
            ApStreamingWrapperPartialView {
                inner: self.inner.clone(),
                map_view: self.map.view(),
            }
        }
    }

    impl<'a> ApStreamingWrapperView<'a> {
        pub fn can_remove(&self, ix: Ix3) -> bool {
            if let Some(id) = self.map_view[ix] {
                !self.inner.is_ap(id)
            } else {
                true
            }
        }
    }

    impl<'a> ApStreamingWrapperPartialView<'a> {
        pub fn replace_view(&mut self, transformed: ApStreamingWrapperView<'a>) {
            self.map_view = transformed.map_view;
        }

        // okay to not modify mapping since add_ix should never be requested
        #[inline]
        pub fn connect_node(&mut self, id: NodeId, add_ix: AddIx) {
            for neighbor in neighbor_ixs(add_ix.ix()) {
                if let Some(neighbor_id) = self.map_view.get(neighbor).copied().flatten() {
                    self.inner.add_edge(neighbor_id, id);
                }
            }
        }

        pub fn can_remove(&self, ix: Ix3) -> bool {
            if let Some(id) = self.map_view[ix] {
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

#[inline]
fn is_reflection(transformation: Transformation) -> bool {
    let comb = transformation.comb();
    let perm = transformation.perm();

    let comb = comb.mask();

    // shift x to first position
    let mut perm = perm.indicies();
    let x_pos = perm.iter().position(|&p| p == 0).unwrap();
    perm.rotate_left(x_pos);

    // check for odd number of reflections
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
#[cfg(not(feature = "output"))]
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
