#![engine(cuda::engine)]

use cuda::dmem::{Buffer, DSend};
use cuda::gpu;

use std::fs::File;
use std::io::prelude::*;


#[derive(Debug)]
pub struct BenchResults {
    pub gpu_time: f64,
    pub test: String,
    pub size: usize,
}

use std::time::Instant;

#[kernel]
unsafe fn matrix_mul(a: &[i32], b: &[i32], mut c: Buffer<i32>, n: i32) {
    let tid = gpu::global_tid_x();
    let i = tid / n;
    let j = tid % n;
    if !(i < n && j < n) {
        return;
    }
    let mut sum = 0;
    let mut a_index = (i * n) as usize;
    let mut b_index = j as usize;
    let c_index = a_index + b_index;
    let mut k = 0;
    let n = n as usize;
    let end_n = n / 4;
    while k < end_n {
        //sum = a[a_index as usize] * b[b_index as usize] + sum;
        sum = a.get_unchecked(a_index) * b.get_unchecked(b_index) + sum;
        a_index += 1;
        b_index += n;
        sum = a.get_unchecked(a_index) * b.get_unchecked(b_index) + sum;
        a_index += 1;
        b_index += n;
        sum = a.get_unchecked(a_index) * b.get_unchecked(b_index) + sum;
        a_index += 1;
        b_index += n;
        sum = a.get_unchecked(a_index) * b.get_unchecked(b_index) + sum;
        a_index += 1;
        b_index += n;
        k += 1;
    }
    for _ in end_n*4..n {
        sum += a.get_unchecked(a_index) * b.get_unchecked(b_index);
        a_index += 1;
        b_index += n;
    }

    c.set(c_index, sum);
    
}

pub fn run_bench1() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();
    let sizes = [16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384];

    for &n in &sizes {
        let total = (n * n) as usize;

        // Generate deterministic pseudo-random matrix data
        let mut a = vec![0; n * n];
        let mut b = vec![0; n * n];
        for i in 0..n * n {
            // random values
            let v1 = (i * 1234567) % 1000;
            let v2 = (i * 7654321) % 1000;
            a[i] = v1 as i32;
            b[i] = v2 as i32;
        }

        let mut d_a = a.as_slice().to_device().unwrap();
        let mut d_b = b.as_slice().to_device().unwrap();
        let buf_c = Buffer::<i32>::alloc(total).unwrap();
        let mut d_buf_c = buf_c.to_device().unwrap();
        let mut d_n = (n as i32).to_device().unwrap();

        let threads_per_block = 64;
        let blocks = (n * n + threads_per_block - 1) / threads_per_block;

        let start = Instant::now();

        matrix_mul
            .launch_with_dptr(threads_per_block, blocks, &mut d_a, &mut d_b, &mut d_buf_c, &mut d_n)
            .unwrap();


        let x = buf_c.retrieve().unwrap();
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        // println!(
        //     "Matrix size: {}x{} | GPU: {:.3} ms",
        //     n,
        //     n,
        //     elapsed_gpu_ms
        // );

        results.push(BenchResults {
            gpu_time: elapsed_gpu_ms,
            test: format!("{}x{}", n, n),
            size: n,
        });
    }

    results
}

pub fn run_bench2() -> Vec<BenchResults> {
    match matrix_mul.pre_compile() {
        Ok(_) => {},
        Err(e) => {
            println!("Kernel compilation failed: {:?}", e);
            panic!("Kernel compilation failed");
        }
    }

    let mut results = Vec::new();

    let n = 1024;
    let total = n * n;
    let iters = [1, 10, 20, 50, 100, 200, 500];

    for &iter in &iters {
        // Generate deterministic pseudo-random matrix data
        let mut a = vec![0; n * n];
        let mut b = vec![0; n * n];
        for i in 0..n * n {
            // random values
            let v1 = (i * 1234567) % 1000;
            let v2 = (i * 7654321) % 1000;
            a[i] = v1 as i32;
            b[i] = v2 as i32;
        }

        let mut d_a = a.as_slice().to_device().unwrap();
        let mut d_b = b.as_slice().to_device().unwrap();
        let mut buf_c = Buffer::<i32>::alloc(total).unwrap().to_device().unwrap();

        let mut d_n = (n as i32).to_device().unwrap();

        let threads_per_block = 64;
        let blocks = (total + threads_per_block - 1) / threads_per_block;

        let start = Instant::now();

        for _ in 0..iter {
            matrix_mul
                .launch_with_dptr(threads_per_block, blocks, &mut d_a, &mut d_b, &mut buf_c, &mut d_n)
                .unwrap();
            // GPU sync is implicit in launch for many wrappers. If needed: gpu::sync().unwrap();
        }


        let x = buf_c.retrieve().unwrap();
        let elapsed_gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchResults {
            gpu_time: elapsed_gpu_ms,
            test: format!("{}x{} - {} iters", n, n, iter),
            size: iter,
        });
    }

    results
    
}



fn main() {
    // Run the benchmarks
    println!("Running benchmarks...");
    //println!("Running Bench 1...");
    //let results1 = run_bench1();
    println!("Running Bench 2...");
    let results2 = run_bench2();

    // Print the results
    //println!("Bench 1 Results:");
    // for result in &results1 {
    //     println!("Test: {} | GPU Time: {:.3} ms", result.test, result.gpu_time);
    // }

    println!("\nBench 2 Results:");
    for result in &results2 {
        println!("Test: {} | GPU Time: {:.3} ms", result.test, result.gpu_time);
    }

    // write the results to a file
    // in a format that can be read by python
    // such as json
    let mut res = "".to_string();
    
    // for result in &results1 {
    //     //writeln!(res, "B1|{}|{}", result.test, result.gpu_time).unwrap();
    //     res.push_str(&format!("B1|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    // }
    for result in &results2 {
        //writeln!(res, "B2|{}|{}", result.test, result.gpu_time).unwrap();
        res.push_str(&format!("B2|{}|{}|{}\n", result.test, result.size, result.gpu_time));
    }
    let mut file = File::create("bench_results.txt").unwrap();
    file.write_all(res.as_bytes()).unwrap();
    println!("Benchmarks complete. Results written to bench_results.txt");
}
