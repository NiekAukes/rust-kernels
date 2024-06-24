use std::{marker::Tuple, rc::Rc, sync::Arc};

extern crate lazy_static;
use lazy_static::lazy_static;
use crate::{dmem::{DPtr, DSend}, kernel::CudaDim, module::Module, CUDAError};
use crate::get_cuda;
#[derive(Debug)]
pub struct Kernel<Dim, Args> 
where Args: Tuple{
    pub name: &'static str,
    pub code: &'static [u8],
    pub phantom: std::marker::PhantomData<(Dim, Args)>,
}

// compiler cache
use std::collections::HashMap;
use std::sync::Mutex;

struct Cache {
    cache: HashMap<&'static str, Arc<Module<'static>>>,
}

impl Cache {
    fn get(&self, name: &'static str) -> Option<Arc<Module<'static>>> {
        self.cache.get(name).map(|m| Arc::clone(m))
    }

    fn insert(&mut self, name: &'static str, module: Module<'static>) -> Arc<Module<'static>>{
        self.cache.insert(name, Arc::new(module));
        self.cache.get(name).unwrap().clone()
    }
}

lazy_static! {
    static ref CACHE: Mutex<Cache> = Mutex::new(Cache {
        cache: HashMap::new()
    });
}

pub fn compile_program<'a>(code : &[u8], name: &'static str) -> Result<Arc<Module<'static>>, CUDAError> {
    // check if the module is already compiled
    let mut cache = CACHE.lock().unwrap();
    if let Some(module) = cache.get(name) {
        return Ok(module);
    }
    let program = nvvm::NvvmProgram::new().unwrap();
    match program.add_module(code, name.to_string()) {
        Ok(_) => {},
        Err(e) => println!("Error adding module: {:?}", e),
    }

    let ptx = match program.compile(&[nvvm::NvvmOption::NoOpts]) {
        Ok(c) => c,
        Err(e) => {
            println!("Error compiling: {:?}", e);
            return Err(CUDAError::Unknown(0));
        }
    };

    let cuda = get_cuda();
    // create a module
    let module = cuda.add_module(ptx.as_slice())?;
    // add it to the cache
    Ok(cache.insert(name, module))
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

    pub fn launch_with_dptr(&self, threads_per_block: Dim, blocks: Dim, arg: &mut DPtr<T>) -> Result<(), CUDAError> {
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
