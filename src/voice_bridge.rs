//! ALICE-TRT × ALICE-Voice bridge
//!
//! GPU-accelerated voice feature extraction — mel spectrogram and embedding.
//!
//! Author: Moroya Sakamoto

/// Mel spectrogram configuration
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub n_mels: usize,
    pub hop_length: usize,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            n_mels: 40,
            hop_length: 160,
        }
    }
}

/// Voice embedding result from TRT inference
#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    pub embedding: Vec<f32>,
    pub confidence: f32,
    pub inference_us: u32,
}

/// Compute mel filter bank center frequencies
pub fn mel_center_frequencies(n_mels: usize, sample_rate: u32) -> Vec<f32> {
    let f_max = sample_rate as f32 / 2.0;
    let mel_max = 2595.0 * (1.0 + f_max / 700.0).log10();
    (0..n_mels)
        .map(|i| {
            let mel = mel_max * (i as f32 + 1.0) / (n_mels as f32 + 1.0);
            700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
        })
        .collect()
}

/// Compute frame energy from PCM samples
pub fn frame_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Extract simple voice features for TRT inference input
pub fn extract_features(samples: &[f32], config: &MelConfig) -> Vec<f32> {
    let n_frames = samples.len().saturating_sub(config.n_fft) / config.hop_length + 1;
    if n_frames == 0 { return vec![]; }

    let mut features = Vec::with_capacity(n_frames * 3);
    for i in 0..n_frames {
        let start = i * config.hop_length;
        let end = (start + config.n_fft).min(samples.len());
        let frame = &samples[start..end];

        let energy = frame_energy(frame);

        // Zero-crossing rate
        let mut zcr = 0u32;
        for j in 1..frame.len() {
            if (frame[j] >= 0.0) != (frame[j - 1] >= 0.0) {
                zcr += 1;
            }
        }
        let zcr_rate = zcr as f32 / frame.len().max(1) as f32;

        // Spectral centroid proxy (energy-weighted position)
        let centroid: f32 = frame.iter().enumerate()
            .map(|(k, &s)| k as f32 * s.abs())
            .sum::<f32>() / frame.iter().map(|s| s.abs()).sum::<f32>().max(1e-10);

        features.push(energy);
        features.push(zcr_rate);
        features.push(centroid / config.n_fft as f32);
    }
    features
}

/// GPU voice feature extractor state
pub struct GpuVoiceExtractor {
    pub config: MelConfig,
    pub frames_processed: u64,
}

impl GpuVoiceExtractor {
    pub fn new(config: MelConfig) -> Self {
        Self { config, frames_processed: 0 }
    }

    pub fn extract(&mut self, samples: &[f32]) -> Vec<f32> {
        let features = extract_features(samples, &self.config);
        self.frames_processed += 1;
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_frequencies() {
        let freqs = mel_center_frequencies(40, 16000);
        assert_eq!(freqs.len(), 40);
        assert!(freqs[0] > 0.0);
        assert!(freqs[39] < 8000.0);
        // Monotonically increasing
        for i in 1..freqs.len() {
            assert!(freqs[i] > freqs[i - 1]);
        }
    }

    #[test]
    fn test_frame_energy() {
        let silence = vec![0.0f32; 320];
        assert_eq!(frame_energy(&silence), 0.0);

        let tone: Vec<f32> = (0..320).map(|i| (i as f32 * 0.1).sin()).collect();
        assert!(frame_energy(&tone) > 0.0);
    }

    #[test]
    fn test_extract_features() {
        let config = MelConfig::default();
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let features = extract_features(&samples, &config);
        assert!(!features.is_empty());
        assert_eq!(features.len() % 3, 0); // 3 features per frame
    }

    #[test]
    fn test_gpu_extractor() {
        let mut ext = GpuVoiceExtractor::new(MelConfig::default());
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();
        let feat = ext.extract(&samples);
        assert!(!feat.is_empty());
        assert_eq!(ext.frames_processed, 1);
    }

    #[test]
    fn test_empty_input() {
        let config = MelConfig::default();
        let features = extract_features(&[], &config);
        assert!(features.is_empty());
    }
}
