use find_cuda_helper;

fn main() {
    let paths = find_cuda_helper::find_cuda_lib_dirs();
    for path in paths {
        println!("cargo:rustc-link-search={}", path.display());
    }
    println!("cargo:rustc-link-lib=nvJitLink");
}