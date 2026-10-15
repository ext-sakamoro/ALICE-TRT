//! ALICE-TRT × ALICE-DB bridge
//!
//! Inference telemetry persistence — latency, throughput, and model metrics,
//! stored in the byte key-value (blob) store of [`alice_db::AliceDB`].
//!
//! # Encoding
//!
//! Each [`InferenceRecord`] is one blob entry.
//!
//! - **Key** (24 bytes): `model_hash[16] || timestamp_ms.to_be_bytes()[8]`.
//!   The timestamp is big-endian so that the byte-lexicographic order of keys
//!   that share a model hash is the numeric order of `timestamp_ms`.
//!   `timestamp_ms` is a `u64`, so no sign bit has to be flipped: the
//!   big-endian bytes of an unsigned integer sort exactly like the integer
//!   (`0` first, `u64::MAX` last).
//! - **Value** (34 bytes, [`InferenceRecord::to_bytes`]): `model_hash[16] ||
//!   timestamp_ms (u64 LE) || latency_us (u32 LE) || batch_size (u16 LE) ||
//!   throughput_fps (f32 LE bits)`.
//!
//! A query for one model is [`alice_db::AliceDB::scan_blob_prefix`] on the
//! 16-byte model hash (keys come back in ascending byte order, hence ascending
//! `timestamp_ms`), followed by the timestamp filter `timestamp_ms >= from_ms`.
//! Entries under the prefix whose key is not 24 bytes long are not records of
//! this bridge and are skipped; a 24-byte key whose value does not decode, or
//! decodes to a different model hash / timestamp than the key, is reported as
//! [`std::io::ErrorKind::InvalidData`].
//!
//! Author: Moroya Sakamoto

use std::io;

use alice_db::AliceDB;

/// Length of the blob key: 16-byte model hash + 8-byte big-endian timestamp.
pub const KEY_LEN: usize = 24;

/// Length of the serialized [`InferenceRecord`].
pub const RECORD_LEN: usize = 34;

/// Inference metric record
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRecord {
    pub model_hash: [u8; 16],
    pub timestamp_ms: u64,
    pub latency_us: u32,
    pub batch_size: u16,
    pub throughput_fps: f32,
}

impl InferenceRecord {
    /// Serialize to 34-byte binary
    pub fn to_bytes(&self) -> [u8; RECORD_LEN] {
        let mut buf = [0u8; RECORD_LEN];
        buf[0..16].copy_from_slice(&self.model_hash);
        buf[16..24].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf[24..28].copy_from_slice(&self.latency_us.to_le_bytes());
        buf[28..30].copy_from_slice(&self.batch_size.to_le_bytes());
        buf[30..34].copy_from_slice(&self.throughput_fps.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < RECORD_LEN {
            return None;
        }
        let mut model_hash = [0u8; 16];
        model_hash.copy_from_slice(&buf[0..16]);
        Some(Self {
            model_hash,
            timestamp_ms: u64::from_le_bytes(buf[16..24].try_into().ok()?),
            latency_us: u32::from_le_bytes(buf[24..28].try_into().ok()?),
            batch_size: u16::from_le_bytes(buf[28..30].try_into().ok()?),
            throughput_fps: f32::from_le_bytes(buf[30..34].try_into().ok()?),
        })
    }

    /// Blob key of this record (see the module documentation).
    pub fn key(&self) -> [u8; KEY_LEN] {
        record_key(&self.model_hash, self.timestamp_ms)
    }
}

/// Blob key for `model_hash` at `timestamp_ms`: the hash followed by the
/// big-endian timestamp.
pub fn record_key(model_hash: &[u8; 16], timestamp_ms: u64) -> [u8; KEY_LEN] {
    let mut key = [0u8; KEY_LEN];
    key[0..16].copy_from_slice(model_hash);
    key[16..24].copy_from_slice(&timestamp_ms.to_be_bytes());
    key
}

/// Timestamp encoded in a 24-byte record key.
fn key_timestamp(key: &[u8]) -> Option<u64> {
    if key.len() != KEY_LEN {
        return None;
    }
    Some(u64::from_be_bytes(key[16..24].try_into().ok()?))
}

fn invalid_data(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// TRT inference metrics DB store
pub struct TrtDbStore {
    db: AliceDB,
    pub total_records: u64,
}

impl TrtDbStore {
    pub fn new(db: AliceDB) -> Self {
        Self {
            db,
            total_records: 0,
        }
    }

    /// Store an inference record keyed by model_hash + timestamp.
    ///
    /// A second record with the same model hash and timestamp replaces the
    /// first.
    ///
    /// # Errors
    /// Returns the error of [`AliceDB::put_blob`] (blob WAL append failure).
    pub fn store_record(&mut self, record: &InferenceRecord) -> io::Result<()> {
        self.db.put_blob(&record.key(), &record.to_bytes())?;
        self.total_records += 1;
        Ok(())
    }

    /// Records of `model_hash` with `timestamp_ms >= from_ms` (inclusive, no
    /// upper bound: `u64::MAX` is included), in ascending `timestamp_ms`.
    ///
    /// # Errors
    /// Returns the error of [`AliceDB::scan_blob_prefix`], or
    /// [`io::ErrorKind::InvalidData`] when a record entry does not decode or
    /// does not match its key.
    pub fn query_by_model(
        &self,
        model_hash: &[u8; 16],
        from_ms: u64,
    ) -> io::Result<Vec<InferenceRecord>> {
        let mut out = Vec::new();
        for (key, value) in self.db.scan_blob_prefix(model_hash)? {
            let Some(ts) = key_timestamp(&key) else {
                continue;
            };
            if ts < from_ms {
                continue;
            }
            let record = InferenceRecord::from_bytes(&value).ok_or_else(|| {
                invalid_data(format!(
                    "db_bridge: record value of {} bytes, expected {RECORD_LEN}",
                    value.len()
                ))
            })?;
            if record.model_hash != *model_hash || record.timestamp_ms != ts {
                return Err(invalid_data(format!(
                    "db_bridge: record (timestamp {}) does not match its key (timestamp {ts})",
                    record.timestamp_ms
                )));
            }
            out.push(record);
        }
        Ok(out)
    }

    /// Average latency for a model over the records with
    /// `timestamp_ms >= from_ms`; `None` when there are none.
    ///
    /// # Errors
    /// See [`Self::query_by_model`].
    pub fn avg_latency_us(&self, model_hash: &[u8; 16], from_ms: u64) -> io::Result<Option<f64>> {
        let records = self.query_by_model(model_hash, from_ms)?;
        if records.is_empty() {
            return Ok(None);
        }
        let sum: u64 = records.iter().map(|r| r.latency_us as u64).sum();
        Ok(Some(sum as f64 / records.len() as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "alice-trt-db-bridge-{tag}-{}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn open_store(dir: &TempDir) -> TrtDbStore {
        TrtDbStore::new(AliceDB::open(&dir.0).unwrap())
    }

    fn rec(model: u8, ts: u64, latency_us: u32) -> InferenceRecord {
        InferenceRecord {
            model_hash: [model; 16],
            timestamp_ms: ts,
            latency_us,
            batch_size: (ts % 7) as u16 + 1,
            throughput_fps: latency_us as f32 * 0.5,
        }
    }

    fn timestamps(records: &[InferenceRecord]) -> Vec<u64> {
        records.iter().map(|r| r.timestamp_ms).collect()
    }

    #[test]
    fn test_record_serialization() {
        let record = InferenceRecord {
            model_hash: [0xAB; 16],
            timestamp_ms: 1_700_000_000_000,
            latency_us: 1500,
            batch_size: 8,
            throughput_fps: 120.5,
        };
        let bytes = record.to_bytes();
        let restored = InferenceRecord::from_bytes(&bytes).unwrap();
        assert_eq!(restored.model_hash, [0xAB; 16]);
        assert_eq!(restored.latency_us, 1500);
        assert_eq!(restored.batch_size, 8);
        assert!((restored.throughput_fps - 120.5).abs() < 0.01);
    }

    #[test]
    fn test_record_too_short() {
        assert!(InferenceRecord::from_bytes(&[0; 10]).is_none());
    }

    #[test]
    fn key_layout_is_hash_then_big_endian_timestamp() {
        let key = record_key(&[0x11; 16], 0x0102_0304_0506_0708);
        assert_eq!(&key[..16], &[0x11; 16]);
        assert_eq!(&key[16..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(key_timestamp(&key), Some(0x0102_0304_0506_0708));
    }

    #[test]
    fn round_trip_is_bit_equal_after_reopen() {
        let dir = TempDir::new("roundtrip");
        let original = InferenceRecord {
            model_hash: *b"0123456789abcdef",
            timestamp_ms: 1_700_000_000_123,
            latency_us: 4321,
            batch_size: 17,
            throughput_fps: f32::from_bits(0x42F1_0000),
        };
        {
            let mut store = open_store(&dir);
            store.store_record(&original).unwrap();
            assert_eq!(store.total_records, 1);
            let got = store.query_by_model(&original.model_hash, 0).unwrap();
            assert_eq!(got, vec![original.clone()]);
            store.db.close().unwrap();
        }
        let store = open_store(&dir);
        let raw = store.db.get_blob(&original.key()).unwrap().unwrap();
        assert_eq!(raw, original.to_bytes().to_vec());
        let got = store.query_by_model(&original.model_hash, 0).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].to_bytes(), original.to_bytes());
        assert_eq!(
            got[0].throughput_fps.to_bits(),
            original.throughput_fps.to_bits()
        );
    }

    #[test]
    fn prefix_isolates_models() {
        let dir = TempDir::new("isolation");
        let mut store = open_store(&dir);
        // model B's hash differs from A's only in the last byte, so its keys
        // sort right after A's
        let a = [0x10; 16];
        let mut b = a;
        b[15] = 0x11;
        for ts in [5u64, 1, 9] {
            store
                .store_record(&InferenceRecord {
                    model_hash: a,
                    ..rec(0, ts, 100)
                })
                .unwrap();
            store
                .store_record(&InferenceRecord {
                    model_hash: b,
                    ..rec(0, ts + 1, 200)
                })
                .unwrap();
        }
        let ra = store.query_by_model(&a, 0).unwrap();
        let rb = store.query_by_model(&b, 0).unwrap();
        assert_eq!(timestamps(&ra), vec![1, 5, 9]);
        assert_eq!(timestamps(&rb), vec![2, 6, 10]);
        assert!(ra.iter().all(|r| r.model_hash == a && r.latency_us == 100));
        assert!(rb.iter().all(|r| r.model_hash == b && r.latency_us == 200));
        assert_eq!(store.avg_latency_us(&a, 0).unwrap(), Some(100.0));
        assert_eq!(store.avg_latency_us(&b, 0).unwrap(), Some(200.0));
    }

    #[test]
    fn ascending_timestamp_order_including_edges() {
        let dir = TempDir::new("order");
        let mut store = open_store(&dir);
        // inserted out of order; 1 / 256 and 255 / 65536 reverse under a
        // little-endian key, and u64::MAX / 1 << 63 test the top bit
        let inserted = [
            u64::MAX,
            256,
            0,
            1 << 63,
            1,
            65_536,
            255,
            (1 << 63) - 1,
            u64::MAX - 1,
        ];
        for ts in inserted {
            store.store_record(&rec(7, ts, ts as u32)).unwrap();
        }
        let got = store.query_by_model(&[7; 16], 0).unwrap();
        let mut want = inserted.to_vec();
        want.sort_unstable();
        assert_eq!(timestamps(&got), want);
        for r in &got {
            assert_eq!(*r, rec(7, r.timestamp_ms, r.timestamp_ms as u32));
        }
    }

    #[test]
    fn from_ms_is_an_inclusive_lower_bound() {
        let dir = TempDir::new("bounds");
        let mut store = open_store(&dir);
        for ts in [9u64, 10, 11, u64::MAX] {
            store.store_record(&rec(3, ts, ts as u32)).unwrap();
        }
        let m = [3; 16];
        assert_eq!(
            timestamps(&store.query_by_model(&m, 10).unwrap()),
            vec![10, 11, u64::MAX]
        );
        assert_eq!(
            timestamps(&store.query_by_model(&m, 11).unwrap()),
            vec![11, u64::MAX]
        );
        assert_eq!(
            timestamps(&store.query_by_model(&m, 0).unwrap()),
            vec![9, 10, 11, u64::MAX]
        );
        assert_eq!(
            timestamps(&store.query_by_model(&m, u64::MAX).unwrap()),
            vec![u64::MAX]
        );
        // 9 + 10 + 11 + 0xFFFF_FFFF (u64::MAX as u32), over 4
        assert_eq!(
            store.avg_latency_us(&m, 9).unwrap(),
            Some((9 + 10 + 11 + u64::from(u32::MAX)) as f64 / 4.0)
        );
    }

    #[test]
    fn empty_ranges() {
        let dir = TempDir::new("empty");
        let mut store = open_store(&dir);
        let m = [4; 16];
        assert!(store.query_by_model(&m, 0).unwrap().is_empty());
        assert_eq!(store.avg_latency_us(&m, 0).unwrap(), None);
        for ts in [1u64, 2, 3] {
            store.store_record(&rec(4, ts, 50)).unwrap();
        }
        assert!(store.query_by_model(&m, 4).unwrap().is_empty());
        assert!(store.query_by_model(&m, u64::MAX).unwrap().is_empty());
        assert_eq!(store.avg_latency_us(&m, 4).unwrap(), None);
        assert!(store.query_by_model(&[5; 16], 0).unwrap().is_empty());
    }

    #[test]
    fn same_key_overwrites() {
        let dir = TempDir::new("overwrite");
        let mut store = open_store(&dir);
        store.store_record(&rec(6, 42, 1)).unwrap();
        store.store_record(&rec(6, 42, 2)).unwrap();
        let got = store.query_by_model(&[6; 16], 0).unwrap();
        assert_eq!(got, vec![rec(6, 42, 2)]);
        assert_eq!(store.total_records, 2);
    }

    #[test]
    fn foreign_and_corrupt_entries() {
        let dir = TempDir::new("foreign");
        let mut store = open_store(&dir);
        let m = [8; 16];
        store.store_record(&rec(8, 1, 10)).unwrap();
        // a key under the prefix that is not 24 bytes is not a record
        store.db.put_blob(&[8; 20], b"other").unwrap();
        // longer than a record key is not a record either
        store.db.put_blob(&[8; 30], b"longer").unwrap();
        assert_eq!(timestamps(&store.query_by_model(&m, 0).unwrap()), vec![1]);
        // a record key whose value does not decode is an error, not a skip
        store.db.put_blob(&record_key(&m, 2), &[0; 10]).unwrap();
        let err = store.query_by_model(&m, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        // a value whose timestamp disagrees with its key is an error
        store.db.delete_blob(&record_key(&m, 2)).unwrap();
        store
            .db
            .put_blob(&record_key(&m, 3), &rec(8, 4, 10).to_bytes())
            .unwrap();
        let err = store.query_by_model(&m, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        // a value whose model hash disagrees with its key is an error too
        store.db.delete_blob(&record_key(&m, 3)).unwrap();
        store
            .db
            .put_blob(&record_key(&m, 5), &rec(9, 5, 10).to_bytes())
            .unwrap();
        let err = store.query_by_model(&m, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
