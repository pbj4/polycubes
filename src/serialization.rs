#![allow(dead_code)]

pub mod polycube {
    // format: [x, y, z, ..packed bits]: [u8]

    use ndarray::{Array3, ArrayView3, Dimension};

    pub fn serialize(view: ArrayView3<bool>) -> Vec<u8> {
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

    pub fn deserialize(slice: &[u8]) -> Array3<bool> {
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
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let array = random_array(&mut rng);

            let serialized = serialize(array.view());
            let deserialized = deserialize(&serialized);

            assert_eq!(array, deserialized);
        }
    }

    #[cfg(test)]
    pub fn random_array(rng: &mut rand::rngs::ThreadRng) -> Array3<bool> {
        use rand::Rng;
        let xyz = rng
            .gen::<[std::num::NonZeroU8; 3]>()
            .map(|nz| nz.get().into());
        let vec = std::iter::repeat_with(|| -> bool { rng.gen() })
            .take(xyz.iter().product())
            .collect();
        Array3::from_shape_vec(xyz, vec).unwrap()
    }
}

pub mod job {
    // format: target_n: u8 | polycube

    pub fn serialize(polycube: &[u8], target_n: usize) -> Vec<u8> {
        let mut out = vec![target_n.try_into().unwrap()];
        out.extend_from_slice(polycube);
        out
    }

    pub fn deserialize(slice: &[u8]) -> (usize, &[u8]) {
        let (&n, polycube) = slice.split_first().unwrap();
        (n.into(), polycube)
    }

    #[test]
    fn test_serialization() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let array = super::polycube::random_array(&mut rng);
            let n: usize = rng.gen::<std::num::NonZeroU8>().get().into();

            let serialized_array = super::polycube::serialize(array.view());

            let serialized = serialize(&serialized_array, n);
            let (deserialized_n, deserialized_array) = deserialize(&serialized);

            assert_eq!(n, deserialized_n);
            assert_eq!(&serialized_array, deserialized_array);
        }
    }
}

pub mod result {
    // format: len polycube: u64 be | base polycube | [..(rotation count, reflection count)]: [(u64 be, u64 be)]

    use {ndarray::ArrayView3, std::collections::BTreeMap};

    pub fn serialize(view: ArrayView3<bool>, map: &BTreeMap<usize, (usize, usize)>) -> Vec<u8> {
        let mut polycube = super::polycube::serialize(view);
        let mut counts = serialize_counts(map.iter().enumerate().map(|(i, (k, v))| {
            assert_eq!(i + 1, *k);
            *v
        }));

        let len: u64 = polycube.len().try_into().unwrap();
        let mut out = len.to_be_bytes().to_vec();
        out.append(&mut polycube);
        out.append(&mut counts);
        out
    }

    pub fn serialize_counts(counts: impl Iterator<Item = (usize, usize)>) -> Vec<u8> {
        counts
            .flat_map(|(r, p)| {
                let (r, p): (u64, u64) = (r.try_into().unwrap(), p.try_into().unwrap());
                r.to_be_bytes().into_iter().chain(p.to_be_bytes())
            })
            .collect()
    }

    pub fn as_key_value(slice: &[u8]) -> (&[u8], &[u8]) {
        let (len, data) = slice.split_at(8);
        let len = u64::from_be_bytes(len.try_into().unwrap());
        data.split_at(len.try_into().unwrap())
    }

    pub fn deserialize_counts(counts: &[u8]) -> Vec<(usize, usize)> {
        let bytes_to_usize = |s: &[u8]| -> usize {
            u64::from_be_bytes(s.try_into().unwrap())
                .try_into()
                .unwrap()
        };

        counts
            .chunks_exact(16)
            .map(|c| (bytes_to_usize(&c[0..8]), bytes_to_usize(&c[8..16])))
            .collect()
    }

    #[test]
    fn test_serialization() {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let array = super::polycube::random_array(&mut rng);
            let map: BTreeMap<usize, (usize, usize)> =
                (1..rng.gen::<std::num::NonZeroU8>().get().into())
                    .map(|k| (k, rng.gen()))
                    .collect();

            let serialized = serialize(array.view(), &map);
            let (polycube, counts) = as_key_value(&serialized);

            let deserialized_polycube = super::polycube::deserialize(polycube);
            assert_eq!(array, deserialized_polycube);

            let deserialized_counts: BTreeMap<usize, (usize, usize)> = deserialize_counts(counts)
                .into_iter()
                .enumerate()
                .map(|(k, v)| (k + 1, v))
                .collect();
            assert_eq!(map, deserialized_counts);
        }
    }
}

use nanoserde::{DeBin, SerBin};

#[derive(SerBin, DeBin, PartialEq, Eq, Hash)]
pub struct SerPolycube(pub Vec<u8>);

#[derive(SerBin, DeBin)]
pub struct Results {
    pub counts: Vec<(usize, usize)>,
}

#[derive(SerBin, DeBin)]
pub struct SerResults(pub Vec<u8>);

#[derive(SerBin, DeBin)]
pub struct JobRequest {
    pub jobs_wanted: usize,
    pub results: std::collections::HashMap<SerPolycube, SerResults>,
}

#[derive(SerBin, DeBin)]
pub struct JobResponse {
    pub target_n: usize,
    pub jobs: Vec<SerPolycube>,
}

impl SerPolycube {
    pub fn ser(view: ndarray::ArrayView3<bool>) -> Self {
        use ndarray::Dimension;

        let mut out = vec![0; 3 + (view.len() - 1) / 8 + 1];

        let (header, data) = out.split_at_mut(3);

        header.copy_from_slice(
            &Into::<[usize; 3]>::into(view.raw_dim().into_pattern()).map(|d| d.try_into().unwrap()),
        );

        for (i, &bit) in view.iter().enumerate() {
            data[i / 8] |= (bit as u8) << (i % 8);
        }

        Self(out)
    }

    pub fn de(&self) -> ndarray::Array3<bool> {
        let (header, data) = self.0.split_at(3);
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

        ndarray::Array3::from_shape_vec(xyz, vec).unwrap()
    }
}

impl SerResults {
    pub fn ser(results: &Results) -> Self {
        Self(results.serialize_bin())
    }

    pub fn de(&self) -> Results {
        Results::deserialize_bin(&self.0).unwrap()
    }
}
