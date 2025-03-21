#![engine(cuda::engine)]

use cuda::dmem::{Buffer, DSend};
use cuda::gpu;

#[kernel]
unsafe fn gpu64(mut a: Buffer<i32>) {
    let i = gpu::global_tid_x();
    a.set(i as usize, i);
}

fn main() {
    let threads_per_block = 64;
    let blocks = 4;
    let a = Buffer::<i32>::alloc(threads_per_block * blocks).unwrap();

    // launch the kernel with 64 * 4 threads
    gpu64.launch(threads_per_block, blocks, a).unwrap();

    let result = a.retrieve().unwrap();
    println!("Result: {:?}", result);

    // Test with dptr
    with_dptr();
}

#[kernel]
unsafe fn gpu_cpy(mut a: Buffer<i32>, b: &[i32]) {
    let i = gpu::global_tid_x() as usize;
    let bi = b[i];
    a.set(i, bi);
}

fn with_dptr() {
    let threads_per_block = 10;
    let blocks = 1;
    let a = Buffer::<i32>::alloc(threads_per_block * blocks).unwrap();
    let b = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11];

    let mut a_device = a.to_device().unwrap();
    let mut b_device = b.as_slice().to_device().unwrap();

    // launch the kernel with 64 * 4 threads
    gpu_cpy
        .launch_with_dptr(threads_per_block, blocks, &mut a_device, &mut b_device)
        .unwrap();

    let result = a.retrieve().unwrap();
    println!("Result: {:?}", result);
}
