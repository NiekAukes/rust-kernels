#![feature(lang_items)]
#![feature(tuple_trait)]
#![engine(cuda::engine)]

use core::panic;
use std::panic::PanicInfo;
use std::str;
use std::any::Any;

use cuda::dmem::Buffer;


#[lang = "kernel_exchange_malloc_fn"]
#[no_mangle]
fn kernel_exchange_malloc(size: usize, _align: usize) -> *mut u8 {
    return std::ptr::null_mut();
}

#[lang = "kernel_panic_impl"]
#[no_mangle]
fn kernel_panic_impl(info: &PanicInfo) -> ! {
    unsafe { cuda::gpu::__trap(); }
    loop {}
}

#[lang = "kernel_panic_fmt_impl"]
#[no_mangle]
fn kernel_panic_fmt_impl(info: &PanicInfo) -> ! {
    unsafe { cuda::gpu::__trap(); }
    loop {}
}

trait DoIt {
    fn do_sth(&self) -> i32;
}

struct Thing;
struct Thang;

impl DoIt for Thing {
    fn do_sth(&self) -> i32 {
        421
    }
}

impl DoIt for Thang {
    fn do_sth(&self) -> i32 {
        24
    }
}

#[kernel]
unsafe fn gpu64(mut a: Buffer<i32>) {
    // let tid = cuda::gpu::global_tid_x();
    // if tid != 0  {
    //     return;
    // }
    // let my_thing = Thing;
    // let my_thang = Thang;
    // let trait_obj: &dyn DoIt = &my_thing;
    // let trait_obj2: &dyn DoIt = &my_thang;


    // let _result = trait_obj.do_sth();
    // let _result2 = trait_obj2.do_sth();

    // a.set(0, _result);
    // a.set(1, _result2)
    let _ = Box::new(42) as Box<dyn Any>;
    //panic!("This is a panic from the kernel");
}

fn main() {
    let s = match str::from_utf8(gpu64.code) {
        Ok(v) => v,
        Err(e) => panic!("Invalid UTF-8 sequence: {}", e),
    };

    //println!("code: {}", s);

    let mut buffer = Buffer::<i32>::alloc(2).unwrap();

    gpu64.launch(10, 10, buffer);

    let result = buffer.retrieve().unwrap();
    println!("Result {:?}", result);
}