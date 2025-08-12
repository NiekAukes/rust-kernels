use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::Mutex;

use crate::kernel::Kernel;
use crate::CUDAError;
use crate::ToResult;
use crate::CUDA;
use crate::sys;

#[derive(Debug, Clone)]
pub struct Module {
    cuda: Arc<Mutex<CUDA>>,
    pub(crate) _module: sys::CUmodule,
}
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub fn new(cuda: &Arc<Mutex<CUDA>>, cubin: &[u8]) -> Result<Self, CUDAError> {
        let mut module = MaybeUninit::uninit();
        let image = cubin.as_ptr() as *const std::ffi::c_void;
        unsafe { sys::cuModuleLoadData(module.as_mut_ptr(), image).to_result()? };
        Ok(Self {
            cuda: Arc::clone(cuda),
            _module: unsafe { module.assume_init() },
        })
    }

    pub fn get_kernel(&self, name: &str) -> Result<Kernel, CUDAError> {
        Kernel::new(&self, name)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self._module) };
    }
}