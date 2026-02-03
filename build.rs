//! Build script for ALICE-TRT.
//!
//! When the `cuda` feature is enabled, compiles `csrc/bitnet_kernel.cu`
//! into a static library that Rust links against.
//!
//! Requirements (cuda feature only):
//! - NVIDIA CUDA Toolkit (nvcc on PATH or CUDA_TOOLKIT_ROOT_DIR set)
//! - cc crate

fn main() {
    #[cfg(feature = "cuda")]
    {
        build_cuda();
    }
}

#[cfg(feature = "cuda")]
fn build_cuda() {
    let mut build = cc::Build::new();

    build
        .cuda(true)
        .file("csrc/bitnet_kernel.cu")
        .include("csrc")
        // CUDA architectures: Pascal (6.1), Volta (7.0), Turing (7.5),
        //                     Ampere (8.0/8.6), Ada (8.9), Hopper (9.0)
        .flag("-gencode=arch=compute_61,code=sm_61")
        .flag("-gencode=arch=compute_70,code=sm_70")
        .flag("-gencode=arch=compute_75,code=sm_75")
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_86,code=sm_86")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("--expt-relaxed-constexpr")
        .flag("-O3")
        .warnings(false);

    // Allow override via environment variable
    if let Ok(cuda_root) = std::env::var("CUDA_TOOLKIT_ROOT_DIR") {
        build.include(format!("{cuda_root}/include"));
        println!("cargo:rustc-link-search=native={cuda_root}/lib64");
    }

    build.compile("alice_trt_cuda");

    // Link CUDA runtime
    println!("cargo:rustc-link-lib=cudart");

    // Rerun if CUDA source changes
    println!("cargo:rerun-if-changed=csrc/bitnet_kernel.cu");
    println!("cargo:rerun-if-changed=csrc/alice_trt_c_api.h");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
}
