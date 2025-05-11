/*#[kernel]
fn add(a: i32, b: i32) -> i32 {
    a + b
}*/

/// Built-in function for getting the kernel's thread ID.
/// will return 0 if the function is called outside of a kernel.
//extern { fn tid() -> usize; }

//#[kernel]
unsafe fn add_one1(a: &mut [i32]) {
    //let i = __nvvm_thread_idx_x() + 5;
    let i = 0;
    if i < a.len() as i32 {
        a[i as usize] = i + 1;
    }
}

// #[kernel]
// unsafe fn matrix_mul(a: &'static [i32], b: &'static [i32], c: &'static mut [i32], n: i32) {
//     let i = __nvvm_thread_idx_x();
//     let j = __nvvm_thread_idx_y();
//     if i < n && j < n {
//         let mut sum = 0;
//         for k in 0..n {
//             let a_index = i * n + k;
//             let b_index = k * n + j;
//             sum += a[a_index as usize] * b[b_index as usize];
//         }
//         let c_index = i * n + j;
//         c[c_index as usize] = sum;
//     }
// }

// fn main() {
//     let matrix_mul = &add_one1;
//     let code = String::from_utf8(matrix_mul.code.to_vec()).unwrap();
//     //println!("{:?}", gpu64);
//     //println!("{}", code);
//     //println!("{}", gpu64.get_dimension_type());
//     //gpu64.prepare(10).run(&[1.0, 2.0, 3.0]);
//     println!("array size: {:?}", std::mem::size_of::<&[i32]>());

//     // match matrix_mul.pre_compile() {
//     //     Ok(_) => println!("Precompiled successfully"),
//     //     Err(e) => {
//     //         println!("Compilation failed: {:?}", e);
//     //         //return;
//     //     }
//     // }

//     // // put the code in a file
//     // let mut file = std::fs::File::create("gpu64.ll").unwrap();

//     // file.write_all(code.as_bytes()).unwrap();

//     //     let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
//     //     let b = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];
//     //     let mut c = vec![0; 9];

//     //     //matrix_mul.launch(9, 1, &a, &b, &mut c, 3);
//     //     matrix_mul.launch(9, 1, &mut c);
//     //     println!("{:?}", c);
// }

fn main() {
    unsafe {
        let mut a = vec![0; 10];
        add_one1(a.as_mut_slice());
    }
}
