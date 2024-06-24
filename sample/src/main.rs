#![engine(cuda::engine)]

use cuda::dmem::{DSend, DSendOwned};

#[kernel]
fn gpu64(a: &'static mut [i32]) {
    let i = cuda::thread_idx_x();
    a[i] = i as i32;
}

fn main() {
    //....
    
    

    
    gpu64.pre_compile().unwrap();
    println!("Kernel compiled successfully");
    // ==============================================================================
    // ==========================  EXECUTION OF THE KERNEL ==========================
    // ==============================================================================

    let max = 256;
    let mut a: Vec<i32> = vec![1; max as usize];

    let threads_per_block = 64;
    let blocks_per_grid = max / threads_per_block;
    
    match gpu64.launch(threads_per_block, blocks_per_grid, a.as_mut_slice()) {
        Ok(_) => println!("Kernel launched successfully"),
        Err(e) => println!("Error launching kernel: {:?}", e),
    }

    println!("Result: {:?}", a);
    


    // ========================= Example with DBox =========================
    // move a to the device, a is not accessible anymore on the host
    let mut device_a = a.to_device_boxed().unwrap();
    // launch the kernel, launch_with_dptr is NOT blocking, 
    // CPU can do whatever it wants while GPU is working
    match gpu64.launch_with_dptr(threads_per_block, blocks_per_grid, device_a.as_mut_slice()) {
        Ok(_) => println!("Kernel launched successfully"),
        Err(e) => println!("Error launching kernel: {:?}", e),
    }
    // get the result, this is a blocking operation
    let res = device_a.consume().unwrap();
    println!("Result: {:?}", res);

    println!("Benching...");
    bench();
}

fn ave(a: i32, b: i32) -> i32{
    a + b / 2
}

fn gpu64_imitation(a: &mut [i32]) {
    for i in 0..a.len() - 1 {
        a[i] = ave(i as i32, i as i32);
    }

}

fn bench() {
    let sizes = [256, 256 * 256, 256*256*96, 256*256*256];
    
    for i in sizes {
        println!("------------------------------------");
        println!("Benching with size: {}", i);
        let max = i;
        let a: Vec<i32> = vec![1; max as usize];
        let mut d_a = a.to_device_boxed().unwrap();

        let threads_per_block = 256;
        let blocks_per_grid = max / threads_per_block;

        let timer = std::time::Instant::now();
        
        for _ in 0..100 {
            match gpu64.launch_with_dptr(threads_per_block, blocks_per_grid, d_a.as_mut_slice()) {
                Ok(_) => {},
                Err(e) => println!("Error launching kernel: {:?}", e),
            }
        }

        let mut a = d_a.consume().unwrap();

        println!("GPU time: {:?}", timer.elapsed());
        
        let timer = std::time::Instant::now();
        for _ in 0..100 {
            gpu64_imitation(a.as_mut_slice());
        }
        println!("CPU time: {:?}", timer.elapsed());
    }
}