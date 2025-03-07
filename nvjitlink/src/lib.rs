#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![warn(non_upper_case_globals)]
#![allow(warnings, clippy::warnings)] // TODO: remove this

use std::{fmt::{self, Display}, mem::MaybeUninit, os::raw::c_void};
mod sys;


pub trait ToNVLinkInputType {
    fn to_nvlink_input_type(&self) -> sys::nvJitLinkInputType;
}

pub trait ToNVJitErrorResult {
    fn to_result(&self) -> Result<(), NVJitError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NVJitError {
    UnrecognizedOption = 1,
    MissingArch = 2,
    InvalidInput = 3,
    PTXCompile = 4,
    NVVMCompile = 5,
    Internal = 6,
    Threadpool = 7,
    UnrecognizedInput = 8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NVLinkInputType {
    NVJITLINK_INPUT_NONE = 0,
    NVJITLINK_INPUT_CUBIN = 1,
    NVJITLINK_INPUT_PTX = 2,
    NVJITLINK_INPUT_LTOIR = 3,
    NVJITLINK_INPUT_FATBIN = 4,
    NVJITLINK_INPUT_OBJECT = 5,
    NVJITLINK_INPUT_LIBRARY = 6,
    NVJITLINK_INPUT_ANY = 10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    SM_30,
    SM_32,
    SM_35,
    SM_37,
    SM_50,
    SM_52,
    SM_53,
    SM_60,
    SM_61,
    SM_62,
    SM_70,
    SM_72,
    SM_75,
    SM_80,
    SM_86,
    SM_90
}

impl Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arch::SM_30 => write!(f, "sm_30"),
            Arch::SM_32 => write!(f, "sm_32"),
            Arch::SM_35 => write!(f, "sm_35"),
            Arch::SM_37 => write!(f, "sm_37"),
            Arch::SM_50 => write!(f, "sm_50"),
            Arch::SM_52 => write!(f, "sm_52"),
            Arch::SM_53 => write!(f, "sm_53"),
            Arch::SM_60 => write!(f, "sm_60"),
            Arch::SM_61 => write!(f, "sm_61"),
            Arch::SM_62 => write!(f, "sm_62"),
            Arch::SM_70 => write!(f, "sm_70"),
            Arch::SM_72 => write!(f, "sm_72"),
            Arch::SM_75 => write!(f, "sm_75"),
            Arch::SM_80 => write!(f, "sm_80"),
            Arch::SM_86 => write!(f, "sm_86"),
            Arch::SM_90 => write!(f, "sm_90"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NVCompilerOption {
    Architecture(String),
    KernelsUsed(String),
    //TODO!
}

impl Display for NVCompilerOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NVCompilerOption::Architecture(arch) => write!(f, "-arch={}", arch),
            NVCompilerOption::KernelsUsed(kernels) => write!(f, "-kernels-used={}", kernels),
        }
    }
}

impl ToNVJitErrorResult for sys::nvJitLinkResult {
    fn to_result(&self) -> Result<(), NVJitError> {
        match *self {
            sys::nvJitLinkResult_NVJITLINK_SUCCESS => Ok(()),
            sys::nvJitLinkResult_NVJITLINK_ERROR_UNRECOGNIZED_OPTION => Err(NVJitError::UnrecognizedOption),
            sys::nvJitLinkResult_NVJITLINK_ERROR_MISSING_ARCH => Err(NVJitError::MissingArch),
            sys::nvJitLinkResult_NVJITLINK_ERROR_INVALID_INPUT => Err(NVJitError::InvalidInput),
            sys::nvJitLinkResult_NVJITLINK_ERROR_PTX_COMPILE => Err(NVJitError::PTXCompile),
            sys::nvJitLinkResult_NVJITLINK_ERROR_NVVM_COMPILE => Err(NVJitError::NVVMCompile),
            sys::nvJitLinkResult_NVJITLINK_ERROR_INTERNAL => Err(NVJitError::Internal),
            sys::nvJitLinkResult_NVJITLINK_ERROR_THREADPOOL => Err(NVJitError::Threadpool),
            sys::nvJitLinkResult_NVJITLINK_ERROR_UNRECOGNIZED_INPUT => Err(NVJitError::UnrecognizedInput),
            _ => panic!("Unknown error code"),
        }
    }
}

impl ToNVLinkInputType for NVLinkInputType {
    fn to_nvlink_input_type(&self) -> sys::nvJitLinkInputType {
        *self as sys::nvJitLinkInputType
    }
}

pub struct NVJitCompiler {
    handle: sys::nvJitLinkHandle,
}

impl NVJitCompiler {
    pub fn new(options: &[NVCompilerOption]) -> Result<Self, NVJitError> {
        let mut raw = MaybeUninit::uninit();
        let len = options.len();
        println!("{:?}", options.iter().map(|x| format!("{}", x)).collect::<Vec<_>>());
        let mut options = {
            let mut v = vec![];
            for opt in options {
                let s = format!("{}", opt);
                println!("s: {:?}", s);
                let c = std::ffi::CString::new(s).unwrap();
                v.push(c);
            }
            v
        };
        let opt_handle = options.iter().map(|x| x.as_ptr()).collect::<Vec<_>>().as_mut_ptr();
        

        let handle = unsafe { 
            sys::__nvJitLinkCreate_12_5(
                raw.as_mut_ptr(), 
                len as u32,
                opt_handle).to_result()?;
            raw.assume_init()
        };
        
        Ok(NVJitCompiler { handle })
    }

    pub fn add_data(&self, data: &[u8], name: &str, input_type: NVLinkInputType) -> Result<(), NVJitError> {
        let size = data.len();
        let data = data.as_ptr();
        let input_type = input_type as sys::nvJitLinkInputType;
        let name = std::ffi::CString::new(name).unwrap();
        unsafe {
            sys::__nvJitLinkAddData_12_5(self.handle, input_type, data as _, size, name.as_ptr()).to_result()
        }
    }

    pub fn compile(&self) -> Result<Vec<u8>, NVJitError> {
        let mut size = MaybeUninit::uninit();
        unsafe {
            // link the data
            sys::__nvJitLinkComplete_12_5(self.handle).to_result()?;
            sys::__nvJitLinkGetLinkedCubinSize_12_5(self.handle, size.as_mut_ptr()).to_result()?;
            let size = size.assume_init();
            println!("size: {:?}", size);
            // allocate memory for the cubin
            let mut binary_image_container = vec![0u8; size];
            let binary_image = binary_image_container.as_mut_ptr() as *mut c_void;
            sys::__nvJitLinkGetLinkedCubin_12_5(self.handle, binary_image).to_result()?;
            Ok(binary_image_container)
        }
    }

    pub fn get_error_log(&self) -> Result<String, NVJitError> {
        let mut size = MaybeUninit::uninit();
        unsafe {
            sys::__nvJitLinkGetErrorLogSize_12_5(self.handle, size.as_mut_ptr()).to_result()?;
            let size = size.assume_init();
            let mut data = Vec::with_capacity(size);
            sys::__nvJitLinkGetErrorLog_12_5(self.handle, data.as_mut_ptr() as *mut _).to_result()?;
            Ok(String::from_utf8(data).unwrap())
        }
    }
}

impl Drop for NVJitCompiler {
    fn drop(&mut self) {
        unsafe {
            let h: *mut sys::nvJitLinkHandle = &mut self.handle;
            sys::__nvJitLinkDestroy_12_5(h);
        }
    }
}