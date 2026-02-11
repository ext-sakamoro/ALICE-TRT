//! ALICE-TRT × ALICE-View bridge
//!
//! Neural upscaling: TRT inference drives resolution reconstruction (DLSS-like).
//!
//! Author: Moroya Sakamoto

/// Upscale quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UpscaleQuality {
    Performance = 0,    // 1/4 res → full
    Balanced = 1,       // 1/2 res → full
    Quality = 2,        // 3/4 res → full
    UltraQuality = 3,   // near-native
}

/// Neural upscale request
#[derive(Debug, Clone)]
pub struct UpscaleRequest {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub quality: UpscaleQuality,
    pub motion_vectors: bool,
}

/// Neural upscale result metadata
#[derive(Debug, Clone)]
pub struct UpscaleResult {
    pub output_width: u32,
    pub output_height: u32,
    pub inference_us: u32,
    pub psnr_estimate: f32,
}

/// Compute the render scale factor for a quality preset
pub fn render_scale(quality: UpscaleQuality) -> f32 {
    match quality {
        UpscaleQuality::Performance => 0.5,
        UpscaleQuality::Balanced => 0.67,
        UpscaleQuality::Quality => 0.77,
        UpscaleQuality::UltraQuality => 0.87,
    }
}

/// Compute internal render resolution for a given output and quality
pub fn internal_resolution(output_w: u32, output_h: u32, quality: UpscaleQuality) -> (u32, u32) {
    let scale = render_scale(quality);
    let w = ((output_w as f32 * scale) as u32).max(1);
    let h = ((output_h as f32 * scale) as u32).max(1);
    (w, h)
}

/// Estimate PSNR from scale factor (empirical model)
pub fn estimate_psnr(quality: UpscaleQuality) -> f32 {
    match quality {
        UpscaleQuality::Performance => 28.0,
        UpscaleQuality::Balanced => 31.0,
        UpscaleQuality::Quality => 34.0,
        UpscaleQuality::UltraQuality => 37.0,
    }
}

/// Neural upscaler state
pub struct NeuralUpscaler {
    pub frames_upscaled: u64,
    pub total_inference_us: u64,
}

impl NeuralUpscaler {
    pub fn new() -> Self {
        Self { frames_upscaled: 0, total_inference_us: 0 }
    }

    /// Process an upscale request (returns metadata; actual pixel data via View)
    pub fn upscale(&mut self, req: &UpscaleRequest) -> UpscaleResult {
        let (iw, ih) = internal_resolution(req.output_width, req.output_height, req.quality);
        // Estimate inference time: ~1us per 1000 output pixels
        let total_pixels = req.output_width as u64 * req.output_height as u64;
        let inference_us = (total_pixels / 1000).max(1) as u32;
        self.frames_upscaled += 1;
        self.total_inference_us += inference_us as u64;
        UpscaleResult {
            output_width: req.output_width,
            output_height: req.output_height,
            inference_us,
            psnr_estimate: estimate_psnr(req.quality),
        }
    }

    pub fn avg_inference_us(&self) -> f64 {
        if self.frames_upscaled == 0 { return 0.0; }
        self.total_inference_us as f64 / self.frames_upscaled as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_resolution() {
        let (w, h) = internal_resolution(1920, 1080, UpscaleQuality::Performance);
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn test_render_scale_ordering() {
        assert!(render_scale(UpscaleQuality::Performance) < render_scale(UpscaleQuality::Balanced));
        assert!(render_scale(UpscaleQuality::Balanced) < render_scale(UpscaleQuality::Quality));
        assert!(render_scale(UpscaleQuality::Quality) < render_scale(UpscaleQuality::UltraQuality));
    }

    #[test]
    fn test_upscaler() {
        let mut up = NeuralUpscaler::new();
        let req = UpscaleRequest {
            input_width: 960,
            input_height: 540,
            output_width: 1920,
            output_height: 1080,
            quality: UpscaleQuality::Performance,
            motion_vectors: true,
        };
        let result = up.upscale(&req);
        assert_eq!(result.output_width, 1920);
        assert_eq!(result.psnr_estimate, 28.0);
        assert_eq!(up.frames_upscaled, 1);
    }

    #[test]
    fn test_psnr_quality_correlation() {
        assert!(estimate_psnr(UpscaleQuality::Performance) < estimate_psnr(UpscaleQuality::UltraQuality));
    }
}
