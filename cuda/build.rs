use find_cuda_helper;

macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn main() {
    let paths = find_cuda_helper::find_cuda_lib_dirs();
    for path in paths {
        p!("Adding cuda library search path: {}", path.display());
        println!("cargo:rustc-link-search={}", path.display());
    }
    //let curoot = find_cuda_helper::find_cuda_root()
    //.expect("Could not find a cuda installation");
    // println!("cargo:warning=Adding cuda library search path: {}", curoot.display());
    println!("cargo:rustc-link-lib=cuda");

    let lib_device = find_cuda_helper::find_cuda_root()
        .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!")
        .join("nvvm")
        .join("libdevice")
        .join("libdevice.10.bc")
        .to_string_lossy()
        .into_owned();
    println!("cargo:rustc-env=LIB_DEVICE={}", lib_device);
}
