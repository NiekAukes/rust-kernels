use core::panic::PanicInfo;
use std::arch::asm;

extern "C" {
    pub fn __nvvm_thread_idx_x() -> i32;
    pub fn __nvvm_thread_idx_y() -> i32;
    pub fn __nvvm_thread_idx_z() -> i32;
    pub fn __nvvm_block_idx_x() -> i32;
    pub fn __nvvm_block_idx_y() -> i32;
    pub fn __nvvm_block_idx_z() -> i32;
    pub fn __nvvm_block_dim_x() -> i32;
    pub fn __nvvm_block_dim_y() -> i32;
    pub fn __nvvm_block_dim_z() -> i32;
    pub fn __nvvm_grid_dim_x() -> i32;
    pub fn __nvvm_grid_dim_y() -> i32;
    pub fn __nvvm_grid_dim_z() -> i32;
    pub fn __nvvm_warp_size() -> i32;
    pub fn __trap();
}

#[inline(always)]
pub fn global_tid_x() -> i32 {
    unsafe {
        let thread_idx_x = __nvvm_thread_idx_x();
        let block_idx_x = __nvvm_block_idx_x();
        let block_dim_x = __nvvm_block_dim_x();
        block_idx_x * block_dim_x + thread_idx_x
    }
}

#[inline(always)]
#[target = "nvvm"]
pub fn syncthreads() {
    unsafe {
        asm!("bar.sync 0;", options(nostack, preserves_flags));
    }
}

#[inline(always)]
#[target = "nvvm"]
pub fn syncwarp() {
    unsafe {
        asm!("bar.warp.sync 0;", options(nostack, preserves_flags));
    }
}

#[lang = "kernel_panic_impl"]
#[inline(always)]
pub fn kernel_panic_impl(info: &PanicInfo) -> ! {
    unsafe {
        crate::gpu::__trap();
    }
    loop {}
}

#[lang = "kernel_panic_fmt_impl"]
#[inline(always)]
pub fn kernel_panic_fmt_impl(info: &PanicInfo) -> ! {
    unsafe {
        crate::gpu::__trap();
    }
    loop {}
}

#[lang = "kernel_panic_nounwind_impl"]
#[inline(always)]
pub fn kernel_panic_nounwind_impl(expr: &'static str) -> ! {
    unsafe {
        crate::gpu::__trap();
    }
    loop {}
}
