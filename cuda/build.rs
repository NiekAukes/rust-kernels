use find_cuda_helper;

fn main() {
    let paths = find_cuda_helper::find_cuda_lib_dirs();
    for path in paths {
        println!("cargo:rustc-link-search={}", path.display());
    }
    //let curoot = find_cuda_helper::find_cuda_root()
            //.expect("Could not find a cuda installation");
   // println!("cargo:warning=Adding cuda library search path: {}", curoot.display());
    println!("cargo:rustc-link-lib=cuda");
}