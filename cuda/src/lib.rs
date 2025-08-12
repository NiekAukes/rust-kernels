#![feature(tuple_trait)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(negative_impls)]
#![feature(lang_items)]
#![feature(asm_experimental_arch)]
#![allow(warnings)] // TODO: remove this

use std::{
    mem::MaybeUninit,
    ptr::{null, null_mut},
    rc::Rc,
    sync::{Arc, Mutex},
};

use nvvm::NvvmError;
pub mod atom;
pub mod dmem;
pub mod engine;
pub mod gpu;
pub mod kernel;
pub mod module;
#[allow(warnings)]
pub mod sys;

#[macro_use]
extern crate lazy_static;

unsafe impl std::marker::Send for CUDA {}

// lazy_static! {
//     static ref cuda: Arc<Mutex<CUDA>> = Arc::new(Mutex::new(CUDA::dummy()));
// }

thread_local! {
    static LOCAL_CUDA: Arc<Mutex<CUDA>> = Arc::new(Mutex::new(CUDA::dummy()));
}

// pub fn get_cuda() -> Arc<Mutex<CUDA>> {
//     let mut c = cuda.lock().unwrap();
//     if c.ctx.is_null() {
//         *c = CUDA::new().expect("Failed to initialize CUDA");
//     }
//     println!("CUDA ctx: {:?}", c.ctx);
//     drop(c); // Explicitly release lock before returning
//     cuda.clone()
// }

pub fn get_cuda() -> Arc<Mutex<CUDA>> {
    LOCAL_CUDA.with(|local_cuda| {
        let mut cuda = local_cuda.lock().unwrap();
        if cuda.ctx.is_null() {
            *cuda = CUDA::new().expect("Failed to initialize CUDA");
        }
        drop(cuda); // Explicitly release lock before returning
        local_cuda.clone()
    })
}

// TODO: add more error handling
// currently this is done on_demand
pub trait ToResult {
    fn to_result(self) -> Result<(), CUDAError>;
}

#[derive(Debug)]
pub enum CUDAError {
    NoDevice,
    DeviceNotFound,
    DeviceNotInitialized,
    InvalidSource,
    IllegalAddress,
    InvalidPtx,
    NVVMError(NvvmError),
    Unknown(sys::cudaError_enum),
}

impl ToResult for sys::cudaError_enum {
    fn to_result(self) -> Result<(), CUDAError> {
        match self {
            sys::cudaError_enum_CUDA_SUCCESS => Ok(()),
            sys::cudaError_enum_CUDA_ERROR_NO_DEVICE => Err(CUDAError::NoDevice),
            sys::cudaError_enum_CUDA_ERROR_NOT_FOUND => Err(CUDAError::DeviceNotFound),
            sys::cudaError_enum_CUDA_ERROR_INVALID_SOURCE => Err(CUDAError::InvalidSource),
            sys::cudaError_enum_CUDA_ERROR_ILLEGAL_ADDRESS => Err(CUDAError::IllegalAddress),
            sys::cudaError_enum_CUDA_ERROR_INVALID_PTX => Err(CUDAError::InvalidPtx),
            _ => Err(CUDAError::Unknown(self)),
        }
    }
}

#[derive(Debug)]
pub struct CUDA {
    pub devices: Vec<Device>,
    ctx: sys::CUcontext,
}

impl CUDA {
    pub fn new() -> Result<Self, CUDAError> {
        // initialize the CUDA runtime
        unsafe { sys::cuInit(0).to_result()? };

        // get the number of devices
        let mut device_count = 0;
        unsafe { sys::cuDeviceGetCount(&mut device_count).to_result()? };

        // get the devices
        let mut devices = Vec::with_capacity(device_count as usize);
        for i in 0..device_count {
            let device = unsafe {
                let mut dp = MaybeUninit::uninit();
                sys::cuDeviceGet(dp.as_mut_ptr(), i).to_result()?;
                let dp = dp.assume_init();
                let mut name = [0u8; 256];
                sys::cuDeviceGetName(name.as_mut_ptr() as *mut i8, 256, dp).to_result()?;
                Device {
                    name: String::from_utf8(name.to_vec()).unwrap(),
                    handle: dp,
                }
            };
            devices.push(device);
        }

        // set the primary device, if there is one
        let ctx = {
            if let Some(device) = devices.first() {
                let mut pctx = MaybeUninit::uninit();
                unsafe {
                    sys::cuCtxCreate_v2(pctx.as_mut_ptr(), 0, device.handle).to_result()?;
                    pctx.assume_init()
                }
            } else {
                return Err(CUDAError::NoDevice);
            }
        };

        Ok(CUDA { devices, ctx })
    }

    // pub fn add_module(&self, module: &[u8]) -> Result<module::Module, CUDAError> {
    //     module::Module::new(self, module)
    // }

    pub fn dummy() -> CUDA {
        CUDA {
            devices: vec![],
            ctx: null_mut(),
        }
    }
}

impl Drop for CUDA {
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }
        unsafe { sys::cuCtxDestroy_v2(self.ctx).to_result().unwrap() };
    }
}

#[derive(Debug)]
pub struct Device {
    name: String,
    handle: sys::CUdevice,
}

pub fn device_sync() -> Result<(), CUDAError> {
    unsafe { sys::cuCtxSynchronize().to_result() }
}
