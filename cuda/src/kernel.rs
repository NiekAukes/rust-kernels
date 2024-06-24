use std::{any::Any, os::raw::c_void, ptr::{self, addr_of}};

use crate::{dmem::{DSend, DPtr}, module::Module, sys::{self, CU_LAUNCH_PARAM_END_AS_INT}, ToResult};

pub struct Kernel<'a> {
    module: &'a Module<'a>,
    pub(crate) name: String,
    pub(crate) _function: sys::CUfunction,
}

impl<'a> Kernel<'a> {
    pub fn new(module: &'a Module<'a>, name: &str) -> Result<Self, crate::CUDAError> {
        let mut function = std::ptr::null_mut();
        let name = std::ffi::CString::new(name).unwrap();
        unsafe { sys::cuModuleGetFunction(&mut function, module._module, name.as_ptr()).to_result()? };
        Ok(Self {
            module,
            name: name.into_string().unwrap(),
            _function: function,
        })
    }

    pub fn module(&self) -> &Module {
        self.module
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn launch(
        &self,
        grid_dim: &dyn CudaDim,
        block_dim: &dyn CudaDim,
        shared_mem_bytes: u32,
        args: &[&DPtr<'_, ()>],
    ) -> Result<(), crate::CUDAError> {
        let mut args = {
            let mut a = vec![];
            for arg in args {
                let b = (&(arg._device_ptr as u64)) as *const _;
                let c = b as *mut c_void;
                a.push(c);
            }
            a
        };
        //let e = [CU_LAUNCH_PARAM_END_AS_INT].as_ptr() as *mut std::ffi::c_void;
        //let mut extraparams = [e; 1];
        unsafe {
            sys::cuLaunchKernel(
                self._function,
                grid_dim.x(),
                grid_dim.y(),
                grid_dim.z(),
                block_dim.x(),
                block_dim.y(),
                block_dim.z(),
                shared_mem_bytes,
                ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                ptr::null_mut(),
            )
            .to_result()?;
        }
        Ok(())
    }
}

pub trait CudaDim {
    fn x(&self) -> u32;
    fn y(&self) -> u32;
    fn z(&self) -> u32;
}

impl CudaDim for (u32, u32, u32) {
    fn x(&self) -> u32 {
        self.0
    }

    fn y(&self) -> u32 {
        self.1
    }

    fn z(&self) -> u32 {
        self.2
    }
}

impl CudaDim for (u32, u32) {
    fn x(&self) -> u32 {
        self.0
    }

    fn y(&self) -> u32 {
        self.1
    }

    fn z(&self) -> u32 {
        1
    }
}

impl CudaDim for u32 {
    fn x(&self) -> u32 {
        *self
    }

    fn y(&self) -> u32 {
        1
    }

    fn z(&self) -> u32 {
        1
    }
}


impl CudaDim for (i32, i32, i32) {
    fn x(&self) -> u32 {
        self.0 as u32
    }

    fn y(&self) -> u32 {
        self.1 as u32
    }

    fn z(&self) -> u32 {
        self.2 as u32
    }
}

impl CudaDim for (i32, i32) {
    fn x(&self) -> u32 {
        self.0 as u32
    }

    fn y(&self) -> u32 {
        self.1 as u32
    }

    fn z(&self) -> u32 {
        1
    }
}

impl CudaDim for i32 {
    fn x(&self) -> u32 {
        *self as u32
    }

    fn y(&self) -> u32 {
        1
    }

    fn z(&self) -> u32 {
        1
    }
}

impl CudaDim for usize {
    fn x(&self) -> u32 {
        *self as u32
    }

    fn y(&self) -> u32 {
        1
    }

    fn z(&self) -> u32 {
        1
    }
}