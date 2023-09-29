#![allow(dead_code)]

// format: [x, y, z, ..packed bits]: [u8]

use ndarray::{Array3, ArrayView3, Dimension};

pub fn serialize_view(view: ArrayView3<bool>) -> Vec<u8> {
    let mut out = vec![0; 3 + (view.len() - 1) / 8 + 1];

    let (header, data) = out.split_at_mut(3);

    header.copy_from_slice(
        &Into::<[usize; 3]>::into(view.raw_dim().into_pattern()).map(|d| d.try_into().unwrap()),
    );

    for (i, &bit) in view.iter().enumerate() {
        data[i / 8] |= (bit as u8) << (i % 8);
    }

    out
}

pub fn deserialize_slice(slice: &[u8]) -> Array3<bool> {
    let (header, data) = slice.split_at(3);
    let xyz = TryInto::<[u8; 3]>::try_into(header)
        .unwrap()
        .map(|u| u.into());

    let mut vec = Vec::with_capacity(data.len() * 8);

    for byte in data {
        vec.extend_from_slice(&[
            byte & 0b00000001 != 0,
            byte & 0b00000010 != 0,
            byte & 0b00000100 != 0,
            byte & 0b00001000 != 0,
            byte & 0b00010000 != 0,
            byte & 0b00100000 != 0,
            byte & 0b01000000 != 0,
            byte & 0b10000000 != 0,
        ]);
    }

    vec.truncate(xyz.iter().product());

    Array3::from_shape_vec(xyz, vec).unwrap()
}

#[test]
fn test_serialization() {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let xyz = rng
            .gen::<[std::num::NonZeroU8; 3]>()
            .map(|nz| nz.get().into());
        let vec = std::iter::repeat_with(|| -> bool { rng.gen() })
            .take(xyz.iter().product())
            .collect();

        let array = Array3::from_shape_vec(xyz, vec).unwrap();

        let serialized = serialize_view(array.view());
        let deserialized = deserialize_slice(&serialized);

        assert_eq!(array, deserialized);
    }
}
