use find_cuda_helper::find_libnvvm_bin_dir;

macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn main() {
    p!("searched for nvvm in: {}", find_libnvvm_bin_dir());
    println!("cargo:rustc-link-search={}", find_libnvvm_bin_dir());
    println!("cargo:rustc-link-lib=nvvm");
}
