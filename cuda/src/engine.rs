use std::cell::RefCell;
use std::io::Write;
use std::{marker::Tuple, rc::Rc, sync::Arc};

use std::env;

extern crate lazy_static;
use crate::{
    dmem::{DPtr, DSend},
    kernel::CudaDim,
    module::Module,
    CUDAError,
};
use crate::{get_cuda, kernel};
use lazy_static::lazy_static;
use std::panic::PanicInfo;
#[derive(Debug)]
pub struct Kernel<Dim, Args>
where
    Args: Tuple,
{
    pub name: &'static str,
    pub code: &'static [u8],
    pub phantom: std::marker::PhantomData<(Dim, Args)>,
}

// compiler cache
use std::collections::HashMap;
use std::sync::Mutex;

struct Cache {
    cache: HashMap<&'static str, Arc<Module>>,
}

impl Cache {
    fn get(&self, name: &'static str) -> Option<Arc<Module>> {
        self.cache.get(name).map(|m| Arc::clone(m))
    }

    fn insert(&mut self, name: &'static str, module: Module) -> Arc<Module> {
        self.cache.insert(name, Arc::new(module));
        self.cache.get(name).unwrap().clone()
    }
}


// program-wide storage for compiled PTX code from NVVM
lazy_static! {
    static ref PTXCACHE: Mutex<HashMap<&'static str, Vec<u8>>> = Mutex::new(HashMap::new());
}

// thread-local storage for loaded modules. Modules can not be shared between threads
thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache {
        cache: HashMap::new()
    });
}

pub fn compile_program<'a>(
    code: &[u8],
    name: &'static str,
) -> Result<Arc<Module>, CUDAError> {
    // check if the module is already compiled
    let mut present = CACHE.with(|c| c.borrow().get(name));
    if let Some(module) = present {
        return Ok(module);
    }

    // check if another thread already compiled this module
    let mut cache = PTXCACHE.lock().unwrap();
    let ptx = match cache.get(name) {
        Some(ptx) => ptx.clone(),
        None => {
            // if not, compile it
            let program = nvvm::NvvmProgram::new().unwrap();
            match program.add_module(code, name.to_string()) {
                Ok(_) => {}
                Err(e) => {
                    println!("Error adding nvvm module: {:?}", e);
                    return Err(CUDAError::NVVMError(e));
                }
            }

            let ptx = match program.compile(&[
                nvvm::NvvmOption::FmaContraction,
                nvvm::NvvmOption::Arch(nvvm::NvvmArch::Compute75),
            ]) {
                Ok(c) => c,
                Err(e) => {
                    println!("Error compiling: {:?}", nvvm::get_error_string(e));
                    // print the compiler log
                    let log = program.compiler_log().unwrap();
                    if let Some(log) = log {
                        println!("Compiler log: {:?}", log);
                    }
                    return Err(CUDAError::Unknown(0));
                }
            };
        

            if (env::var("CUDA_DEBUG").is_ok()) {
                // // encode the PTX as a string
                let ptxs = String::from_utf8(ptx.clone()).unwrap();
                //println!("PTX: {}", ptxs);
                let mut file = std::fs::File::create("gpu64.ptx").unwrap();

                file.write_all(&ptx).unwrap();

                println!("PTX written to gpu64.ptx");
            }
            ptx
        }
    };

    let cuda = get_cuda();
    // create a module
    let module = match Module::new(&cuda, ptx.as_slice()) {
        Ok(m) => m,
        Err(e) => {
            println!("Error adding module: {:?}", e);
            return Err(e);
        }
    };

    // add it to the cache
    cache.insert(name, ptx);
    let module = Arc::new(module);
    CACHE.with(|mut c| c.borrow_mut().cache.insert(name, module.clone()));
    Ok(module)
}

pub fn load_kernel_from_ptx(
    ptx: &[u8],
    name: &'static str,
) -> Result<Arc<Module>, CUDAError> {
    let cuda = get_cuda();
    // create a module
    let module = match Module::new(&cuda, ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("Error adding module: {:?}", e);
            return Err(e);
        }
    };

    // add it to the cache
    let module = Arc::new(module);
    CACHE.with(|mut c| c.borrow_mut().cache.insert(name, module.clone()));
    Ok(module)
}

impl<Dim: CudaDim, Args: Tuple> Kernel<Dim, Args> {
    pub fn pre_compile(&self) -> Result<(), CUDAError> {
        compile_program(self.code, self.name)?;
        Ok(())
    }
}

impl<Dim: CudaDim> Kernel<Dim, ()> {
    /// Basic kernel launcher, no arguments, does not block
    pub fn launch(&self, threads_per_block: Dim, blocks: Dim) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}", args);

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(&threads_per_block, &blocks, 0, &[])?;
        Ok(())
    }
}

impl<Dim: CudaDim, T: DSend> Kernel<Dim, (T,)> {
    /// Kernel launcher with one argument located on the host, blocks until results have been received
    pub fn launch(&self, threads_per_block: Dim, blocks: Dim, arg: T) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}", args);
        let mut arg = arg;
        let mut t1 = arg.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(&blocks, &threads_per_block, 0, &[t1.pass()])?;

        // retrieve the result
        arg.copy_from_device(t1)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg: &mut DPtr<T>,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}", args);
        //let mut t1 = arg.to_device()?;
        // compile the program
        let module = compile_program(self.code, self.name).unwrap();
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(&blocks, &threads_per_block, 0, &[arg.pass()])?;

        // retrieve the result
        //arg.copy_from_device(t1).unwrap();
        Ok(())
    }
}

impl<Dim: CudaDim, T: DSend, U: DSend> Kernel<Dim, (T, U)> {
    /// Kernel launcher with two arguments located on the host, blocks until results have been received
    pub fn launch(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: T,
        arg2: U,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}", arg1, arg2);
        let mut arg1 = arg1;
        let mut arg2 = arg2;
        let mut t1 = arg1.to_device()?;
        let mut t2 = arg2.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(&blocks, &threads_per_block, 0, &[t1.pass(), t2.pass()])?;

        // retrieve the result
        arg1.copy_from_device(t1)?;
        arg2.copy_from_device(t2)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: &mut DPtr<T>,
        arg2: &mut DPtr<U>,
    ) -> Result<(), CUDAError> {
        let module = compile_program(self.code, self.name).unwrap();
        let kernel = module.get_kernel(self.name)?;
        kernel.launch(&blocks, &threads_per_block, 0, &[arg1.pass(), arg2.pass()])?;

        Ok(())
    }
}

impl<Dim: CudaDim, T: DSend, U: DSend, V: DSend> Kernel<Dim, (T, U, V)> {
    /// Kernel launcher with three arguments located on the host, blocks until results have been received
    pub fn launch(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: T,
        arg2: U,
        arg3: V,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}", arg1, arg2, arg3);
        let mut arg1 = arg1;
        let mut arg2 = arg2;
        let mut arg3 = arg3;
        let mut t1 = arg1.to_device()?;
        let mut t2 = arg2.to_device()?;
        let mut t3 = arg3.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[t1.pass(), t2.pass(), t3.pass()],
        )?;

        // retrieve the result
        arg1.copy_from_device(t1)?;
        arg2.copy_from_device(t2)?;
        arg3.copy_from_device(t3)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: &mut DPtr<T>,
        arg2: &mut DPtr<U>,
        arg3: &mut DPtr<V>,
    ) -> Result<(), CUDAError> {
        let module = compile_program(self.code, self.name).unwrap();
        let kernel = module.get_kernel(self.name)?;
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[arg1.pass(), arg2.pass(), arg3.pass()],
        )?;

        Ok(())
    }
}

impl<Dim: CudaDim, T: DSend, U: DSend, V: DSend, W: DSend> Kernel<Dim, (T, U, V, W)> {
    /// Kernel launcher with four arguments located on the host, blocks until results have been received
    pub fn launch(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: T,
        arg2: U,
        arg3: V,
        arg4: W,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut arg1 = arg1;
        let mut arg2 = arg2;
        let mut arg3 = arg3;
        let mut arg4 = arg4;
        let mut t1 = arg1.to_device()?;
        let mut t2 = arg2.to_device()?;
        let mut t3 = arg3.to_device()?;
        let mut t4 = arg4.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[t1.pass(), t2.pass(), t3.pass(), t4.pass()],
        )?;

        // retrieve the result
        arg1.copy_from_device(t1)?;
        arg2.copy_from_device(t2)?;
        arg3.copy_from_device(t3)?;
        arg4.copy_from_device(t4)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: &mut DPtr<T>,
        arg2: &mut DPtr<U>,
        arg3: &mut DPtr<V>,
        arg4: &mut DPtr<W>,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut t1 = arg1;
        let mut t2 = arg2;
        let mut t3 = arg3;
        let mut t4 = arg4;

        // compile the program
        let module = compile_program(self.code, self.name).unwrap();
        // get the kernel
        let kernel = module.get_kernel(self.name).unwrap();

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[t1.pass(), t2.pass(), t3.pass(), t4.pass()],
        )?;

        return Ok(());
    }
}

impl<Dim: CudaDim, T: DSend, U: DSend, V: DSend, W: DSend, A: DSend, B: DSend, C: DSend>
    Kernel<Dim, (T, U, V, W, A, B, C)>
{
    /// Kernel launcher with four arguments located on the host, blocks until results have been received
    pub fn launch(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: T,
        arg2: U,
        arg3: V,
        arg4: W,
        arg5: A,
        arg6: B,
        arg7: C,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut arg1 = arg1;
        let mut arg2 = arg2;
        let mut arg3 = arg3;
        let mut arg4 = arg4;
        let mut arg5 = arg5;
        let mut arg6 = arg6;
        let mut arg7 = arg7;
        let mut t1 = arg1.to_device()?;
        let mut t2 = arg2.to_device()?;
        let mut t3 = arg3.to_device()?;
        let mut t4 = arg4.to_device()?;
        let mut t5 = arg5.to_device()?;
        let mut t6 = arg6.to_device()?;
        let mut t7 = arg7.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[
                t1.pass(),
                t2.pass(),
                t3.pass(),
                t4.pass(),
                t5.pass(),
                t6.pass(),
                t7.pass(),
            ],
        )?;

        // retrieve the result
        arg1.copy_from_device(t1)?;
        arg2.copy_from_device(t2)?;
        arg3.copy_from_device(t3)?;
        arg4.copy_from_device(t4)?;
        arg5.copy_from_device(t5)?;
        arg6.copy_from_device(t6)?;
        arg7.copy_from_device(t7)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: &mut DPtr<T>,
        arg2: &mut DPtr<U>,
        arg3: &mut DPtr<V>,
        arg4: &mut DPtr<W>,
        arg5: &mut DPtr<A>,
        arg6: &mut DPtr<B>,
        arg7: &mut DPtr<C>,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut t1 = arg1;
        let mut t2 = arg2;
        let mut t3 = arg3;
        let mut t4 = arg4;
        let mut t5 = arg5;
        let mut t6 = arg6;
        let mut t7 = arg7;

        // compile the program
        let module = compile_program(self.code, self.name).unwrap();
        // get the kernel
        let kernel = module.get_kernel(self.name).unwrap();

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[
                t1.pass(),
                t2.pass(),
                t3.pass(),
                t4.pass(),
                t5.pass(),
                t6.pass(),
                t7.pass(),
            ],
        )?;

        return Ok(());
    }
}

impl<
        Dim: CudaDim,
        T: DSend,
        U: DSend,
        V: DSend,
        W: DSend,
        A: DSend,
        B: DSend,
        C: DSend,
        D: DSend,
    > Kernel<Dim, (T, U, V, W, A, B, C, D)>
{
    /// Kernel launcher with four arguments located on the host, blocks until results have been received
    pub fn launch(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: T,
        arg2: U,
        arg3: V,
        arg4: W,
        arg5: A,
        arg6: B,
        arg7: C,
        arg8: D,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut arg1 = arg1;
        let mut arg2 = arg2;
        let mut arg3 = arg3;
        let mut arg4 = arg4;
        let mut arg5 = arg5;
        let mut arg6 = arg6;
        let mut arg7 = arg7;
        let mut arg8 = arg8;
        let mut t1 = arg1.to_device()?;
        let mut t2 = arg2.to_device()?;
        let mut t3 = arg3.to_device()?;
        let mut t4 = arg4.to_device()?;
        let mut t5 = arg5.to_device()?;
        let mut t6 = arg6.to_device()?;
        let mut t7 = arg7.to_device()?;
        let mut t8 = arg8.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[
                t1.pass(),
                t2.pass(),
                t3.pass(),
                t4.pass(),
                t5.pass(),
                t6.pass(),
                t7.pass(),
                t8.pass(),
            ],
        )?;

        // retrieve the result
        arg1.copy_from_device(t1)?;
        arg2.copy_from_device(t2)?;
        arg3.copy_from_device(t3)?;
        arg4.copy_from_device(t4)?;
        arg5.copy_from_device(t5)?;
        arg6.copy_from_device(t6)?;
        arg7.copy_from_device(t7)?;
        arg8.copy_from_device(t8)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: &mut DPtr<T>,
        arg2: &mut DPtr<U>,
        arg3: &mut DPtr<V>,
        arg4: &mut DPtr<W>,
        arg5: &mut DPtr<A>,
        arg6: &mut DPtr<B>,
        arg7: &mut DPtr<C>,
        arg8: &mut DPtr<D>,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut t1 = arg1;
        let mut t2 = arg2;
        let mut t3 = arg3;
        let mut t4 = arg4;
        let mut t5 = arg5;
        let mut t6 = arg6;
        let mut t7 = arg7;
        let mut t8 = arg8;

        // compile the program
        let module = compile_program(self.code, self.name).unwrap();
        // get the kernel
        let kernel = module.get_kernel(self.name).unwrap();

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[
                t1.pass(),
                t2.pass(),
                t3.pass(),
                t4.pass(),
                t5.pass(),
                t6.pass(),
                t7.pass(),
                t8.pass(),
            ],
        )?;

        return Ok(());
    }
}

impl<
        Dim: CudaDim,
        T: DSend,
        U: DSend,
        V: DSend,
        W: DSend,
        A: DSend,
        B: DSend,
        C: DSend,
        D: DSend,
        E: DSend,
    > Kernel<Dim, (T, U, V, W, A, B, C, D, E)>
{
    /// Kernel launcher with four arguments located on the host, blocks until results have been received
    pub fn launch(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: T,
        arg2: U,
        arg3: V,
        arg4: W,
        arg5: A,
        arg6: B,
        arg7: C,
        arg8: D,
        arg9: E,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut arg1 = arg1;
        let mut arg2 = arg2;
        let mut arg3 = arg3;
        let mut arg4 = arg4;
        let mut arg5 = arg5;
        let mut arg6 = arg6;
        let mut arg7 = arg7;
        let mut arg8 = arg8;
        let mut arg9 = arg9;
        let mut t1 = arg1.to_device()?;
        let mut t2 = arg2.to_device()?;
        let mut t3 = arg3.to_device()?;
        let mut t4 = arg4.to_device()?;
        let mut t5 = arg5.to_device()?;
        let mut t6 = arg6.to_device()?;
        let mut t7 = arg7.to_device()?;
        let mut t8 = arg8.to_device()?;
        let mut t9 = arg9.to_device()?;

        // compile the program
        let module = compile_program(self.code, self.name)?;
        // get the kernel
        let kernel = module.get_kernel(self.name)?;

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[
                t1.pass(),
                t2.pass(),
                t3.pass(),
                t4.pass(),
                t5.pass(),
                t6.pass(),
                t7.pass(),
                t8.pass(),
                t9.pass(),
            ],
        )?;

        // retrieve the result
        arg1.copy_from_device(t1)?;
        arg2.copy_from_device(t2)?;
        arg3.copy_from_device(t3)?;
        arg4.copy_from_device(t4)?;
        arg5.copy_from_device(t5)?;
        arg6.copy_from_device(t6)?;
        arg7.copy_from_device(t7)?;
        arg8.copy_from_device(t8)?;
        arg9.copy_from_device(t9)?;
        Ok(())
    }

    pub fn launch_with_dptr(
        &self,
        threads_per_block: Dim,
        blocks: Dim,
        arg1: &mut DPtr<T>,
        arg2: &mut DPtr<U>,
        arg3: &mut DPtr<V>,
        arg4: &mut DPtr<W>,
        arg5: &mut DPtr<A>,
        arg6: &mut DPtr<B>,
        arg7: &mut DPtr<C>,
        arg8: &mut DPtr<D>,
        arg9: &mut DPtr<E>,
    ) -> Result<(), CUDAError> {
        //println!("Launching kernel {} with {} threads per block and {} blocks", self.name, threads_per_block, blocks);
        //println!("Arguments: {:?}, {:?}, {:?}, {:?}", arg1, arg2, arg3, arg4);
        let mut t1 = arg1;
        let mut t2 = arg2;
        let mut t3 = arg3;
        let mut t4 = arg4;
        let mut t5 = arg5;
        let mut t6 = arg6;
        let mut t7 = arg7;
        let mut t8 = arg8;
        let mut t9 = arg9;

        // compile the program
        let module = compile_program(self.code, self.name).unwrap();
        // get the kernel
        let kernel = module.get_kernel(self.name).unwrap();

        // launch the kernel
        kernel.launch(
            &blocks,
            &threads_per_block,
            0,
            &[
                t1.pass(),
                t2.pass(),
                t3.pass(),
                t4.pass(),
                t5.pass(),
                t6.pass(),
                t7.pass(),
                t8.pass(),
                t9.pass(),
            ],
        )?;

        return Ok(());
    }
}
