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
pub unsafe fn global_tid_x() -> i32 {
    let thread_idx_x = __nvvm_thread_idx_x();
    let block_idx_x = __nvvm_block_idx_x();
    let block_dim_x = __nvvm_block_dim_x();
    //let warp_size = __nvvm_warp_size();
    thread_idx_x + block_idx_x * block_dim_x
}
