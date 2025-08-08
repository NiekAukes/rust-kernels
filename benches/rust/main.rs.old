#![engine(cuda::engine)]
#![feature(tuple_trait)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]

use std::io::Write;
use std::marker::Tuple;
use std::ops::Index;
use std::{arch::asm, marker::PhantomData};

use cuda::dmem::DSend;
use cuda::{engine::Kernel, kernel::CudaDim};

/*#[kernel]
fn add(a: i32, b: i32) -> i32 {
    a + b
}*/

/// Built-in function for getting the kernel's thread ID.
/// will return 0 if the function is called outside of a kernel.
//extern { fn tid() -> usize; }

extern "C" {
    fn __nvvm_thread_idx_x() -> i32;
    fn __nvvm_thread_idx_y() -> i32;
    fn __nvvm_block_idx_x() -> i32;
    fn __nvvm_block_dim_x() -> i32;
    fn __nvvm_grid_dim_x() -> i32;
    fn __nvvm_warp_size() -> i32;
}

#[inline(always)]
unsafe fn get_tid_x() -> i32 {
    let thread_idx_x = __nvvm_thread_idx_x();
    let block_idx_x = __nvvm_block_idx_x();
    let block_dim_x = __nvvm_block_dim_x();
    //let warp_size = __nvvm_warp_size();
    thread_idx_x + block_idx_x * block_dim_x
}

#[kernel]
unsafe fn add_one1(a: &'static mut [i32]) {
    let i = get_tid_x();
    if i < a.len() as i32 {
        a[i as usize] = i + 1;
    }
}

#[kernel]
unsafe fn matrix_mul(a: &[i32], b: &[i32], c: &mut [i32], n: i32) {
    let t = get_tid_x();
    let i = t / n;
    let j = t % n;

    let raw_a = a.as_ptr();
    let raw_b = b.as_ptr();
    //c[t as usize] = n;
    if i < n {
        let mut sum = 0;
        let mut k = 0;
        let itn = i * n;
        while k < n {
            let a_index = itn + k;
            let b_index = k * n + j;
            //let ai = a.get_unchecked(a_index as usize);
            //let bi = b.get_unchecked(b_index as usize);
            //let ai = a[a_index as usize];
            //let bi = b[b_index as usize];
            let ai = *raw_a.offset(a_index as isize);
            let bi = *raw_b.offset(b_index as isize);
            sum += ai * bi;
            k += 1;
        }
        let c_index = i * n + j;
        c[c_index as usize] = sum;
    }
}

#[kernel]
unsafe fn test_intrinsics(a: &mut [i32]) {
    let i = __nvvm_thread_idx_x();
    let j = __nvvm_thread_idx_y();
    let tid = get_tid_x();
    let block_dim_x = __nvvm_block_dim_x();
    let block_idx_x = __nvvm_block_idx_x();

    if i == 0 {
        return;
    }

    a[0] = i;
    a[1] = j;
    a[2] = tid;
    a[3] = block_idx_x;
    a[4] = block_dim_x;
}

fn matrix_mul_cpu(a: &[i32], b: &[i32], c: &mut [i32], n: i32) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0;
            for k in 0..n {
                let a_index = i * n + k;
                let b_index = k * n + j;
                sum += a[a_index as usize] * b[b_index as usize];
            }
            let c_index = i * n + j;
            c[c_index as usize] = sum;
        }
    }
}

fn main() {
    //let matrix_mul = &add_one1;
    let code = String::from_utf8(matrix_mul.code.to_vec()).unwrap();
    //println!("{}", code);
    //println!("{}", gpu64.get_dimension_type());
    //gpu64.prepare(10).run(&[1.0, 2.0, 3.0]);

    match matrix_mul.pre_compile() {
        Ok(_) => println!("Precompiled successfully"),
        Err(e) => {
            println!("Compilation failed: {:?}", e);
            //return;
        }
    }

    // put the code in a file
    let mut file = std::fs::File::create("gpu64.ll").unwrap();

    file.write_all(code.as_bytes()).unwrap();

    // let n: i32 = 10;
    // let a: Vec<i32> = Vec::from_iter((1..=n*n as i32));
    // let b: Vec<i32> = Vec::from_iter((1..=n*n as i32).rev());
    // //let b: Vec<i32> = (n*n+1..1).collect();
    // let mut c = vec![0; (n*n) as usize];

    // println!("a: {:?}", a);
    // println!("b: {:?}", b);

    // match matrix_mul.launch((n*n) as usize, 1, &a, &b, &mut c, n as i32) {
    //     Ok(_) => println!("Launched successfully"),
    //     Err(e) => println!("Launch failed: {:?}", e),
    // }
    // println!("matrix_mux: {:?}", c);
    // // add_one1.launch(9, 1, &mut c);
    // // println!("add_one: {:?}", c);

    // let mut c_cpu = vec![0; (n*n) as usize];
    // matrix_mul_cpu(&a, &b, &mut c_cpu, n);
    // println!("matrix_cpu: {:?}", c_cpu);

    // let mut a = vec![0; 10];
    // test_intrinsics.launch(10, 10, &mut a);
    // println!("test_intrinsics: {:?}", a);
    // println!("test_intrinsics:");
    // println!("nvvm_thread_idx_x: {:?}", a[0]);
    // println!("nvvm_thread_idx_y: {:?}", a[1]);
    // println!("thread_id: {:?}", a[2]);
    // println!("nvvm_block_idx_x: {:?}", a[3]);
    // println!("nvvm_block_dim_x: {:?}", a[4]);

    bench2();
    //bench_simple();
    //run_handcrafted();
}

fn benchcpu() {
    const NS: [usize; 9] = [4, 16, 64, 256, 512, 1024, 2048, 4096, 8192];
    for n in NS {
        // a very large matrix
        //let n = 500;

        println!("Matrix size: {}x{}", n, n);
        let mut a = vec![0; n * n];
        let mut b = vec![0; n * n];
        for i in 0..n * n {
            // random values
            let v1 = (i * 1234567) % 1000;
            let v2 = (i * 7654321) % 1000;
            a[i] = v1 as i32;
            b[i] = v2 as i32;
        }

        let mut c = vec![0; n * n];
        let mut c_cpu = vec![0; n * n];

        let start = std::time::Instant::now();
        matrix_mul_cpu(&a, &b, &mut c_cpu, n as i32);
        let cpu_duration = start.elapsed();

        let threads_per_block = 64;
        let blocks = (n * n + threads_per_block - 1) / threads_per_block;

        let start = std::time::Instant::now();
        match matrix_mul.launch(threads_per_block, blocks, &a, &b, &mut c, n as i32) {
            Ok(_) => (),
            Err(e) => println!("Launch failed: {:?}", e),
        }
        let gpu_duration = start.elapsed();

        //compare the results
        let mut errors = 0;
        for i in 0..n * n {
            if c[i] != c_cpu[i] {
                errors += 1;
                println!("Error at index {}: {} != {}", i, c[i], c_cpu[i]);
            }
        }

        println!("CPU time: {:?}", cpu_duration);
        println!("GPU time: {:?}", gpu_duration);
        //println!("Errors: {}", errors);
    }
}

fn bench1() {
    const NS: [usize; 9] = [4, 16, 64, 256, 512, 1024, 2048, 4096, 8192];
    for n in NS {
        // a very large matrix
        //let n = 500;

        println!("Matrix size: {}x{}", n, n);
        let mut a = vec![0; n * n];
        let mut b = vec![0; n * n];
        for i in 0..n * n {
            // random values
            let v1 = (i * 1234567) % 1000;
            let v2 = (i * 7654321) % 1000;
            a[i] = v1 as i32;
            b[i] = v2 as i32;
        }

        let mut c = vec![0; n * n];
        // let mut c_cpu = vec![0; n * n];

        // let start = std::time::Instant::now();
        // matrix_mul_cpu(&a, &b, &mut c_cpu, n as i32);
        // let cpu_duration = start.elapsed();

        let threads_per_block = 64;
        let blocks = (n * n + threads_per_block - 1) / threads_per_block;

        let start = std::time::Instant::now();
        match matrix_mul.launch(threads_per_block, blocks, &a, &b, &mut c, n as i32) {
            Ok(_) => (),
            Err(e) => println!("Launch failed: {:?}", e),
        }
        let gpu_duration = start.elapsed();

        // compare the results
        // let mut errors = 0;
        // for i in 0..n*n {
        //     if c[i] != c_cpu[i] {
        //         errors += 1;
        //         println!("Error at index {}: {} != {}", i, c[i], c_cpu[i]);
        //     }
        // }

        //println!("CPU time: {:?}", cpu_duration);
        println!("GPU time: {:?}", gpu_duration);
        //println!("Errors: {}", errors);
    }
}

fn bench2() {
    let n = 1024;
    // in this bench, we do a matrix multiplying with itself 1000 times
    // this tests the raw performance of the kernel, and not the memory transfer
    // between the host and the device

    let mut a = vec![0; n * n];
    let mut b = vec![0; n * n];
    for i in 0..n * n {
        // random values
        //let v1 = (i * 1234567) % 1000;
        let v1 = i as i32;
        let v2 = i as i32;
        //let v2 = (i * 7654321) % 1000;
        a[i] = v1 as i32;
        b[i] = v2 as i32;
    }

    let mut c = vec![0; n * n];
    let threads_per_block = 64;
    let blocks = (n * n + threads_per_block - 1) / threads_per_block;

    let mut da = a.as_slice().to_device().unwrap();
    let mut db = b.as_slice().to_device().unwrap();
    let mut dc = c.as_mut_slice().to_device().unwrap();
    let mut dn = (n as i32).to_device().unwrap();

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        matrix_mul
            .launch_with_dptr(
                threads_per_block,
                blocks,
                &mut da,
                &mut db,
                &mut dc,
                &mut dn,
            )
            .unwrap();
    }

    // copy the result back to the host
    let mut d = vec![0; n * n];
    dc.copy_to_host(&mut d).unwrap();

    let gpu_duration = start.elapsed();

    println!("Matrix size: {}x{} - GPU time: {:?}", n, n, gpu_duration);
    println!("first 10 elements: {:?}", &d[..10]);
}

fn bench_simple() {
    let n = 1024;
    let mut a = vec![0; n * n];
    
    let threads_per_block = 64;
    let blocks = (n * n + threads_per_block - 1) / threads_per_block;

    let mut d_a = a.as_mut_slice().to_device().unwrap();

    let start = std::time::Instant::now();
    for _ in 0..10000 {
        match add_one1.launch_with_dptr(threads_per_block, blocks, &mut d_a) {
            Ok(_) => (),
            Err(e) => println!("Launch failed: {:?}", e),
        }
    }

    let mut d = vec![0; n * n];
    d_a.copy_to_host(&mut d).unwrap();

    let gpu_duration = start.elapsed();

    println!("GPU time: {:?}", gpu_duration);
}

const CODE_HC: &[u8] = include_bytes!("../bench2.ptx");

fn run_handcrafted() {
    // load the code from a file
    //let code = std::fs::read_to_string("gpu64.ll").unwrap();
    // let k: Kernel<usize, (*const i32, *const i32, *mut i32, i32)> = Kernel {
    //     name: "_ZN12rust_kernels10matrix_mul17h5d2a1d1469c5ec6cE",
    //     code: CODE_HC.as_bytes(),
    //     phantom: PhantomData::default(),
    // };

    // k.pre_compile();

    let cuda = cuda::get_cuda();
    let module = cuda.add_module(CODE_HC).unwrap();
    let kernel = module.get_kernel("_Z10matrix_mulPKiS0_Pii").unwrap();

    let n = 1024;
    let mut a = vec![0; n * n];
    let mut b = vec![0; n * n];
    for i in 0..n * n {
        // random values
        //let v1 = (i * 1234567) % 1000;
        let v1 = i as i32;
        let v2 = i as i32;
        //let v2 = (i * 7654321) % 1000;
        a[i] = v1 as i32;
        b[i] = v2 as i32;
    }

    let mut c = vec![0; n * n];

    let threads_per_block = 64;
    let blocks = (n * n + threads_per_block - 1) / threads_per_block;

    let mut da = a.as_slice().to_device().unwrap().to_raw_ptr();
    let mut db = b.as_slice().to_device().unwrap().to_raw_ptr();
    let mut dc = c.as_slice().to_device().unwrap();
    let mut dn = (n as i32).to_device().unwrap();

    let dcc = dc.to_raw_ptr();

    kernel
        .launch(
            &threads_per_block,
            &blocks,
            0,
            &[&da.pass(), &db.pass(), &dc.pass(), &dn.pass()],
        )
        .unwrap();

    // copy the result back to the host
    let mut d = vec![0; n * n];
    dc.copy_to_host(&mut d).unwrap();
}
