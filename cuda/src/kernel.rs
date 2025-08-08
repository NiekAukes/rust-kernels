use std::{
    any::Any,
    os::raw::c_void,
    ptr::{self, addr_of},
};

use crate::{
    dmem::{DPassMode, DPtr, DSend},
    module::Module,
    sys::{self, CU_LAUNCH_PARAM_END_AS_INT},
    ToResult,
};

pub struct Kernel<'a> {
    module: &'a Module<'a>,
    pub(crate) name: String,
    pub(crate) _function: sys::CUfunction,
}

impl<'a> Kernel<'a> {
    pub fn new(module: &'a Module<'a>, name: &str) -> Result<Self, crate::CUDAError> {
        let mut function = std::ptr::null_mut();
        let name = std::ffi::CString::new(name).unwrap();
        unsafe {
            sys::cuModuleGetFunction(&mut function, module._module, name.as_ptr()).to_result()?
        };
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
        let mut args_d = vec![];
        
        for arg in args.into_iter() {
            match arg._pass_mode {
                DPassMode::Direct => {
                    let p = std::ptr::addr_of!((*arg)._device_ptr) as *const _;
                    args_d.push(p as *mut c_void);
                }
                DPassMode::Scalar { ref data } => {
                    let p = (data) as *const _;
                    args_d.push(p as *mut c_void);
                }
                DPassMode::Pair { ref data, _size } => {
                    let p1 = std::ptr::addr_of!((*arg)._device_ptr) as *const _;
                    args_d.push(p1 as *mut c_void);

                    let p2 = (data) as *const _;
                    args_d.push(p2 as *mut c_void);
                }
            }
            // let b = std::ptr::addr_of!((*arg)._device_ptr) as *const _;
            // let c = b as *mut c_void;
            // args_d.push(c);
            // if let DPassMode::Pair { data, _size } = arg._pass_mode {
            //     // we have a scalar pair, so pass an extra argument
            //     let d = (&data) as *const _;
            //     let e = d as *mut c_void;
            //     args_d.push(e);
            // }
        }
        //let e = [CU_LAUNCH_PARAM_END_AS_INT].as_ptr() as *mut std::ffi::c_void;
        //let mut extraparams = [e; 1];
        //println!("args_d: {:?}", args_d);

        // try accessing args
        // args is a list of pointers to memory on the host that we want to copy to the device

        // unsafe {
        //     let d = args_d.as_mut_ptr() as *mut *mut c_void;
        //     for i in 0..args_d.len() {
        //         let e = d.add(i);
        //         let f = *e;
        //         let v = *(f as *mut usize);
        //         println!("arg {}: {:?}, value: {:?}", i, f, v);
        //     }
        // }
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
                args_d.as_mut_ptr() as *mut *mut c_void,
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
