//! ALICE-TRT × ALICE-DB bridge
//!
//! Inference telemetry persistence — latency, throughput, and model metrics.
//!
//! Author: Moroya Sakamoto

use alice_db::AliceDB;

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
        if buf.len() < 34 { return None; }
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
        Self { db, total_records: 0 }
    }

    /// Store an inference record keyed by model_hash + timestamp
    pub fn store_record(&mut self, record: &InferenceRecord) {
        let mut key = [0u8; 24];
        key[0..16].copy_from_slice(&record.model_hash);
        key[16..24].copy_from_slice(&record.timestamp_ms.to_be_bytes());
        self.db.put(&key, &record.to_bytes());
        self.total_records += 1;
    }

    /// Query records by model hash and time range
    pub fn query_by_model(&self, model_hash: &[u8; 16], from_ms: u64) -> Vec<InferenceRecord> {
        let mut start = [0u8; 24];
        start[0..16].copy_from_slice(model_hash);
        start[16..24].copy_from_slice(&from_ms.to_be_bytes());
        let mut end = [0u8; 24];
        end[0..16].copy_from_slice(model_hash);
        end[16..24].copy_from_slice(&u64::MAX.to_be_bytes());
        self.db
            .range(&start, &end)
            .filter_map(|(_k, v)| InferenceRecord::from_bytes(&v))
            .collect()
    }

    /// Compute average latency for a model in a time window
    pub fn avg_latency_us(&self, model_hash: &[u8; 16], from_ms: u64) -> Option<f64> {
        let records = self.query_by_model(model_hash, from_ms);
        if records.is_empty() { return None; }
        let sum: u64 = records.iter().map(|r| r.latency_us as u64).sum();
        Some(sum as f64 / records.len() as f64)
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
}
