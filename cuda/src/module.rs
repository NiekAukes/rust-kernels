use std::mem::MaybeUninit;

use crate::kernel::Kernel;
use crate::CUDAError;
use crate::ToResult;
use crate::CUDA;
use crate::sys;

#[derive(Debug, Clone)]
pub struct Module<'a> {
    cuda: &'a CUDA,
    pub(crate) _module: sys::CUmodule,
}
unsafe impl Send for Module<'_> {}
unsafe impl Sync for Module<'_> {}

impl<'a> Module<'a> {
    pub fn new(cuda: &'a CUDA, cubin: &[u8]) -> Result<Self, CUDAError> {
        let mut module = MaybeUninit::uninit();
        let image = cubin.as_ptr() as *const std::ffi::c_void;
        unsafe { sys::cuModuleLoadData(module.as_mut_ptr(), image).to_result()? };
        Ok(Self {
            cuda,
            _module: unsafe { module.assume_init() },
        })
    }

    pub fn cuda(&self) -> &CUDA {
        self.cuda
    }

    pub fn get_kernel(&self, name: &str) -> Result<Kernel, CUDAError> {
        Kernel::new(&self, name)
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        println!("Unloading module");
        unsafe { sys::cuModuleUnload(self._module) };
    }
}