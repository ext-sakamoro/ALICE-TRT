//! ALICE-TRT × ALICE-DB bridge
//!
//! Inference telemetry persistence — latency, throughput, and model metrics.
//!
//! Author: Moroya Sakamoto

use alice_db::AliceDB;
use std::io;

/// Inference metric record
#[derive(Debug, Clone)]
pub struct InferenceRecord {
    pub model_hash: [u8; 16],
    pub timestamp_ms: u64,
    pub latency_us: u32,
    pub batch_size: u16,
    pub throughput_fps: f32,
}

impl InferenceRecord {
    /// Serialize to 34-byte binary
    pub fn to_bytes(&self) -> [u8; 34] {
        let mut buf = [0u8; 34];
        buf[0..16].copy_from_slice(&self.model_hash);
        buf[16..24].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf[24..28].copy_from_slice(&self.latency_us.to_le_bytes());
        buf[28..30].copy_from_slice(&self.batch_size.to_le_bytes());
        buf[30..34].copy_from_slice(&self.throughput_fps.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 34 {
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

    /// Store an inference record keyed by model_hash + timestamp
    ///
    /// # Errors
    ///
    /// Returns the storage error when the blob write fails.
    pub fn store_record(&mut self, record: &InferenceRecord) -> io::Result<()> {
        let mut key = [0u8; 24];
        key[0..16].copy_from_slice(&record.model_hash);
        key[16..24].copy_from_slice(&record.timestamp_ms.to_be_bytes());
        self.db.put_blob(&key, &record.to_bytes())?;
        self.total_records += 1;
        Ok(())
    }

    /// Query records of a model with `timestamp_ms >= from_ms`, in ascending
    /// timestamp order
    ///
    /// # Errors
    ///
    /// Returns the storage error when the blob scan fails.
    pub fn query_by_model(
        &self,
        model_hash: &[u8; 16],
        from_ms: u64,
    ) -> io::Result<Vec<InferenceRecord>> {
        Ok(self
            .db
            .scan_blob_prefix(model_hash)?
            .into_iter()
            .filter_map(|(_k, v)| InferenceRecord::from_bytes(&v))
            .filter(|r| r.timestamp_ms >= from_ms)
            .collect())
    }

    /// Compute average latency for a model in a time window
    ///
    /// # Errors
    ///
    /// Returns the storage error when the blob scan fails.
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

    fn store() -> TrtDbStore {
        let cfg = alice_db::StorageConfig::default();
        TrtDbStore::new(AliceDB::in_memory(cfg).unwrap())
    }

    fn rec(hash: u8, ts: u64, latency_us: u32) -> InferenceRecord {
        InferenceRecord {
            model_hash: [hash; 16],
            timestamp_ms: ts,
            latency_us,
            batch_size: 1,
            throughput_fps: 1.0,
        }
    }

    #[test]
    fn test_store_and_query_round_trip_filters_model_and_time() {
        let mut s = store();
        s.store_record(&rec(1, 300, 30)).unwrap();
        s.store_record(&rec(1, 100, 10)).unwrap();
        s.store_record(&rec(1, 200, 20)).unwrap();
        s.store_record(&rec(2, 150, 999)).unwrap();
        assert_eq!(s.total_records, 4);

        let all = s.query_by_model(&[1; 16], 0).unwrap();
        let ts: Vec<u64> = all.iter().map(|r| r.timestamp_ms).collect();
        assert_eq!(ts, vec![100, 200, 300]);

        let tail = s.query_by_model(&[1; 16], 200).unwrap();
        assert_eq!(tail.len(), 2);
        assert!(s.query_by_model(&[3; 16], 0).unwrap().is_empty());
    }

    #[test]
    fn test_avg_latency_closed_form() {
        let mut s = store();
        assert_eq!(s.avg_latency_us(&[1; 16], 0).unwrap(), None);
        for (ts, l) in [(100, 10), (200, 20), (300, 60)] {
            s.store_record(&rec(1, ts, l)).unwrap();
        }
        assert_eq!(s.avg_latency_us(&[1; 16], 0).unwrap(), Some(30.0));
        assert_eq!(s.avg_latency_us(&[1; 16], 200).unwrap(), Some(40.0));
    }
}
