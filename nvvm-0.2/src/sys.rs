use libloading::{Library, Symbol};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_ulonglong};

use find_cuda_helper::find_cuda_root;
use lazy_static::lazy_static;

pub type NvvmResult = c_int;
pub type NvvmProgram = *mut std::ffi::c_void;
pub type SizeT = c_ulonglong;

pub const nvvmResult_NVVM_SUCCESS: NvvmResult = 0;
pub const nvvmResult_NVVM_ERROR_OUT_OF_MEMORY: NvvmResult = 1;
pub const nvvmResult_NVVM_ERROR_PROGRAM_CREATION_FAILURE: NvvmResult = 2;
pub const nvvmResult_NVVM_ERROR_IR_VERSION_MISMATCH: NvvmResult = 3;
pub const nvvmResult_NVVM_ERROR_INVALID_INPUT: NvvmResult = 4;
pub const nvvmResult_NVVM_ERROR_INVALID_PROGRAM: NvvmResult = 5;
pub const nvvmResult_NVVM_ERROR_INVALID_IR: NvvmResult = 6;
pub const nvvmResult_NVVM_ERROR_INVALID_OPTION: NvvmResult = 7;
pub const nvvmResult_NVVM_ERROR_NO_MODULE_IN_PROGRAM: NvvmResult = 8;
pub const nvvmResult_NVVM_ERROR_COMPILATION: NvvmResult = 9;

#[derive(Debug)]
pub struct Nvvm {
    lib: Library,
    // nvvm_get_error_string: Symbol<'static, unsafe fn(NvvmResult) -> *const c_char>,
    // nvvm_version: Symbol<'static, unsafe fn(*mut c_int, *mut c_int) -> NvvmResult>,
    // nvvm_create_program: Symbol<'static, unsafe fn(*mut NvvmProgram) -> NvvmResult>,
    // nvvm_destroy_program: Symbol<'static, unsafe fn(*mut NvvmProgram) -> NvvmResult>,
}

#[cfg(all(target_os = "windows", not(doc)))]
pub fn find_libnvvm() -> String {
    find_cuda_root()
        .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!")
        .join("nvvm")
        .join("lib")
        .join("x64")
        .join("libnvvm.dll")
        .to_string_lossy()
        .into_owned()
}

#[cfg(all(target_os = "linux", not(doc)))]
pub fn find_libnvvm() -> String {
    find_cuda_root()
        .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!")
        .join("nvvm")
        .join("lib64")
        .join("libnvvm.so")
        .to_string_lossy()
        .into_owned()
}

impl Nvvm {
    // pub fn new(lib_path: &str) -> Result<Self, libloading::Error> {
    //     let lib = unsafe { Library::new(lib_path)? };

    //     unsafe {
    //         Ok(Self {
    //             lib,
    //             nvvm_get_error_string: lib.get(b"nvvmGetErrorString")?,
    //             nvvm_version: lib.get(b"nvvmVersion")?,
    //             nvvm_create_program: lib.get(b"nvvmCreateProgram")?,
    //             nvvm_destroy_program: lib.get(b"nvvmDestroyProgram")?,
    //         })
    //     }
    // }

    pub fn get_error_string(&self, result: NvvmResult) -> String {
        unsafe {
            let func: Symbol<unsafe extern "C" fn(NvvmResult) -> *const c_char> =
                self.lib.get(b"nvvmGetErrorString").unwrap();
            let c_str = func(result);
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }

    pub fn get_version(&self) -> Result<(c_int, c_int), NvvmResult> {
        let mut major = 0;
        let mut minor = 0;
        let result = unsafe {
            let func: Symbol<unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvvmResult> =
                self.lib.get(b"nvvmVersion").unwrap();
            func(&mut major, &mut minor)
        };
        if result == 0 {
            Ok((major, minor))
        } else {
            Err(result)
        }
    }

    pub fn get_ir_version(&self) -> Result<(c_int, c_int), NvvmResult> {
        let mut major = 0;
        let mut minor = 0;
        let mut dbg_major = 0;
        let mut dbg_minor = 0;

        let result = unsafe {
            let func: Symbol<
                unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int, *mut c_int) -> NvvmResult,
            > = self.lib.get(b"nvvmIRVersion").unwrap();
            func(&mut major, &mut minor, &mut dbg_major, &mut dbg_minor)
        };
        if result == 0 {
            Ok((major, minor))
        } else {
            Err(result)
        }
    }

    pub fn get_ir_debug_version(&self) -> Result<(c_int, c_int), NvvmResult> {
        let mut major = 0;
        let mut minor = 0;
        let mut dbg_major = 0;
        let mut dbg_minor = 0;

        let result = unsafe {
            let func: Symbol<
                unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int, *mut c_int) -> NvvmResult,
            > = self.lib.get(b"nvvmIRVersion").unwrap();
            func(&mut major, &mut minor, &mut dbg_major, &mut dbg_minor)
        };
        if result == 0 {
            Ok((dbg_major, dbg_minor))
        } else {
            Err(result)
        }
    }

    pub fn create_program(&self) -> Result<NvvmProgram, NvvmResult> {
        let mut prog: NvvmProgram = std::ptr::null_mut();
        let result = unsafe {
            let func: Symbol<unsafe extern "C" fn(*mut NvvmProgram) -> NvvmResult> =
                self.lib.get(b"nvvmCreateProgram").unwrap();
            func(&mut prog)
        };
        if result == 0 {
            Ok(prog)
        } else {
            Err(result)
        }
    }

    pub fn destroy_program(&self, prog: &mut NvvmProgram) -> Result<(), NvvmResult> {
        let result = unsafe {
            let func: Symbol<unsafe extern "C" fn(*mut NvvmProgram) -> NvvmResult> =
                self.lib.get(b"nvvmDestroyProgram").unwrap();
            func(prog)
        };
        if result == 0 {
            Ok(())
        } else {
            Err(result)
        }
    }

    pub fn compile_program(
        &self,
        prog: NvvmProgram,
        mut options: Vec<*const c_char>,
    ) -> Result<(), NvvmResult> {
        let result = unsafe {
            let func: Symbol<
                unsafe extern "C" fn(
                    NvvmProgram,
                    c_int,
                    *mut *const ::std::os::raw::c_char,
                ) -> NvvmResult,
            > = self.lib.get(b"nvvmCompileProgram").unwrap();
            func(prog, options.len() as c_int, options.as_mut_ptr())
        };
        if result == 0 {
            Ok(())
        } else {
            Err(result)
        }
    }
}

// fn main() {
//     let nvvm = Nvvm::new("libnvvm.so").expect("Failed to load NVVM library");
//     println!("Loaded NVVM library successfully");

//     let version = nvvm.get_version().expect("Failed to get NVVM version");
//     println!("NVVM version: {}.{}", version.0, version.1);
// }

// lazy static the nvvm library
lazy_static! {
    pub static ref NVVM: Nvvm = unsafe {
        let lib_path = find_libnvvm();
        let lib = unsafe { Library::new(lib_path).unwrap() };
        Nvvm { lib }
    };
}

// aliases for the nvvm functions
pub fn nvvmGetErrorString(result: NvvmResult) -> String {
    NVVM.get_error_string(result)
}

pub fn nvvmIRVersion() -> Result<(c_int, c_int), NvvmResult> {
    NVVM.get_ir_version()
}

pub fn nvvmVersion() -> Result<(c_int, c_int), NvvmResult> {
    NVVM.get_version()
}

pub fn nvvmIRDebugVersion() -> Result<(c_int, c_int), NvvmResult> {
    NVVM.get_ir_debug_version()
}

pub fn nvvmCreateProgram() -> Result<NvvmProgram, NvvmResult> {
    NVVM.create_program()
}

pub fn nvvmDestroyProgram(prog: &mut NvvmProgram) -> Result<(), NvvmResult> {
    NVVM.destroy_program(prog)
}

pub unsafe fn nvvmCompileProgram(prog: NvvmProgram, mut options: Vec<*const c_char>) -> NvvmResult {
    let func: Symbol<
        unsafe extern "C" fn(NvvmProgram, c_int, *mut *const ::std::os::raw::c_char) -> NvvmResult,
    > = NVVM.lib.get(b"nvvmCompileProgram").unwrap();
    func(prog, options.len() as c_int, options.as_mut_ptr())
}

pub unsafe fn nvvmGetCompiledResultSize(prog: NvvmProgram, r: *mut SizeT) -> NvvmResult {
    let func: Symbol<unsafe extern "C" fn(NvvmProgram, *mut SizeT) -> NvvmResult> =
        NVVM.lib.get(b"nvvmGetCompiledResultSize").unwrap();
    func(prog, r)
}

pub unsafe fn nvvmGetCompiledResult(prog: NvvmProgram, buf: *mut c_char) -> NvvmResult {
    let func: Symbol<unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResult> =
        NVVM.lib.get(b"nvvmGetCompiledResult").unwrap();
    func(prog, buf)
}

pub unsafe fn nvvmAddModuleToProgram(
    prog: NvvmProgram,
    module: *const c_char,
    size: SizeT,
    name: *const c_char,
) -> NvvmResult {
    let func: Symbol<
        unsafe extern "C" fn(NvvmProgram, *const c_char, SizeT, *const c_char) -> NvvmResult,
    > = NVVM.lib.get(b"nvvmAddModuleToProgram").unwrap();
    func(prog, module, size, name)
}

pub unsafe fn nvvmLazyAddModuleToProgram(
    prog: NvvmProgram,
    module: *const c_char,
    size: SizeT,
    name: *const c_char,
) -> NvvmResult {
    let func: Symbol<
        unsafe extern "C" fn(NvvmProgram, *const c_char, SizeT, *const c_char) -> NvvmResult,
    > = NVVM.lib.get(b"nvvmLazyAddModuleToProgram").unwrap();
    func(prog, module, size, name)
}

pub unsafe fn nvvmGetProgramLogSize(prog: NvvmProgram, r: *mut SizeT) -> NvvmResult {
    let func: Symbol<unsafe extern "C" fn(NvvmProgram, *mut SizeT) -> NvvmResult> =
        NVVM.lib.get(b"nvvmGetProgramLogSize").unwrap();
    func(prog, r)
}

pub unsafe fn nvvmGetProgramLog(prog: NvvmProgram, buf: *mut c_char) -> NvvmResult {
    let func: Symbol<unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResult> =
        NVVM.lib.get(b"nvvmGetProgramLog").unwrap();
    func(prog, buf)
}

pub unsafe fn nvvmVerifyProgram(
    prog: NvvmProgram,
    num_options: SizeT,
    options: *mut *const c_char,
) -> NvvmResult {
    let func: Symbol<unsafe extern "C" fn(NvvmProgram, SizeT, *mut *const c_char) -> NvvmResult> =
        NVVM.lib.get(b"nvvmVerifyProgram").unwrap();
    func(prog, num_options, options)
}
