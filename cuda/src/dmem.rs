use std::{mem::{transmute, MaybeUninit}, os::raw::c_void};

use crate::{sys, CUDAError, ToResult};
use super::CUDA;

// ==================
// ===== Traits =====
// ==================


/// A trait that denotes that this type does not have any references to host memory
/// and can be safely copied to an external device without 
pub trait DeepCopy {}
impl DeepCopy for u8 {}
impl DeepCopy for u16 {}
impl DeepCopy for u32 {}
impl DeepCopy for u64 {}
impl DeepCopy for i8 {}
impl DeepCopy for i16 {}
impl DeepCopy for i32 {}
impl DeepCopy for i64 {}
impl DeepCopy for f32 {}
impl DeepCopy for f64 {}
impl DeepCopy for bool {}
impl DeepCopy for char {}
impl DeepCopy for () {}

pub trait DSend: Sized {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError>;
    fn to_dptr<'a>(&self) -> Result<DPtr<'_, ()>, CUDAError> {
        let dptr = self.to_device()?;
        Ok(unsafe { transmute(dptr) })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError>;
}

pub trait DSendOwned: DSend + ToOwned {
    fn to_device_boxed<'a>(self) -> Result<DBox<'a, Self>, CUDAError> {
        let dptr = self.to_device()?;
        Ok(DBox {
            _inner: dptr,
            _host_data: self,
        })
    }

    fn to_dptr<'a>(self) -> Result<DPtr<'a, ()>, CUDAError> {
        let dptr = self.to_device()?;
        Ok(unsafe { transmute(dptr) })
    }
}

// ================
// ===== DPtr =====
// ================

/// A reference to data on the device
pub struct DPtr<'a, T: DSend> {
    cuda: &'a CUDA,
    pub(crate) _device_ptr: *mut T,
    _size: usize,
}

impl<T: DSend> DPtr<'_, T> {
    pub fn size(&self) -> usize {
        self._size
    }

    pub fn pass(&mut self) -> &mut DPtr<'_, ()> {
        unsafe { transmute(self) }
    }
}

impl<'a, T: DeepCopy> DPtr<'a, Vec<T>> {
    pub fn copy_to_host(&self, container: &mut Vec<T>) -> Result<(), CUDAError> {
        let size = self._size;
        let data = container.as_mut_ptr() as *mut c_void;
        unsafe { 
            sys::cuMemcpyDtoH_v2(data, self._device_ptr as u64, size).to_result()?;
            Ok(())
        }
    }
}

impl<'a, T: DeepCopy> DPtr<'a, &[T]> {
    pub fn copy_to_host(&self, container: &mut [T]) -> Result<(), CUDAError> {
        let size = self._size;
        let data = container.as_mut_ptr() as *mut c_void;
        unsafe { 
            sys::cuMemcpyDtoH_v2(data, self._device_ptr as u64, size).to_result()?;
            Ok(())
        }
    }
}

impl<'a, T: DeepCopy> DPtr<'a, &mut [T]> {
    pub fn copy_to_host(&self, container: &mut [T]) -> Result<(), CUDAError> {
        let size = self._size;
        let data = container.as_mut_ptr() as *mut c_void;
        unsafe { 
            sys::cuMemcpyDtoH_v2(data, self._device_ptr as u64, size).to_result()?;
            Ok(())
        }
    }
}

impl<'a, T: DeepCopy> DPtr<'a, T> {

    pub fn copy_to_host(&self, container: &mut T) -> Result<(), CUDAError> {
        let size = self._size;
        let data = container as *mut T as *mut c_void;
        unsafe { 
            sys::cuMemcpyDtoH_v2(data, self._device_ptr as u64, size).to_result()?;
            Ok(())
        }
    }
}

impl<T: DeepCopy> DSend for T {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
        let cuda = crate::get_cuda();
        let size = std::mem::size_of::<T>();
        let mut dptr = MaybeUninit::uninit();
        let device_ptr = unsafe { 
            sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
            dptr.assume_init() as *mut Self
        };
        unsafe { 
            sys::cuMemcpyHtoD_v2(device_ptr as u64, self as *const T as *const c_void, size).to_result()?;
        }
        Ok(DPtr {
            cuda,
            _device_ptr: device_ptr,
            _size: size,
        })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        dptr.copy_to_host(self)
    }
}

impl<T: DeepCopy> DSend for &[T] {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
        let cuda = crate::get_cuda();
        let size = self.len() * std::mem::size_of::<T>();
        let mut dptr = MaybeUninit::uninit();
        let device_ptr = unsafe { 
            sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
            dptr.assume_init() as u64
        };
        unsafe { 
            sys::cuMemcpyHtoD_v2(device_ptr, self.as_ptr() as *const c_void, size).to_result()?;
        }
        Ok(DPtr {
            cuda,
            _device_ptr: device_ptr as *mut Self,
            _size: size,
        })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        // we can't copy to an immutable slice, return an error
        Err(CUDAError::Unknown(0)) // should not even be possible
    }
}

impl<T: DeepCopy> DSend for &mut [T] {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
        let cuda = crate::get_cuda();
        let size = self.len() * std::mem::size_of::<T>();
        let mut dptr = MaybeUninit::uninit();
        let device_ptr = unsafe { 
            sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
            dptr.assume_init() as u64
        };
        unsafe { 
            sys::cuMemcpyHtoD_v2(device_ptr, self.as_ptr() as *const c_void, size).to_result()?;
        }
        Ok(DPtr {
            cuda,
            _device_ptr: device_ptr as *mut Self,
            _size: size,
        })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        dptr.copy_to_host(self)
    }
}

impl<T: DeepCopy> DSend for Vec<T> {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
        let cuda = crate::get_cuda();
        let size = self.len() * std::mem::size_of::<T>();
        let mut dptr = MaybeUninit::uninit();
        let device_ptr = unsafe { 
            sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
            dptr.assume_init() as u64
        };
        unsafe { 
            sys::cuMemcpyHtoD_v2(device_ptr, self.as_ptr() as *const c_void, size).to_result()?;
        }
        Ok(DPtr {
            cuda,
            _device_ptr: device_ptr as *mut Self,
            _size: size,
        })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        dptr.copy_to_host(self)
    }
}


impl<T: DeepCopy + Clone> DSendOwned for Vec<T> {}

// =================
// ===== DBox ======
// =================

/// A box that is allocated on the device
/// Very similar to `DPtr`, but the box takes ownership of the host data
/// and returns it when it is consumed

pub struct DBox<'a, T: DSend + ToOwned> {
    pub(crate) _inner: DPtr<'a, T>,
    _host_data: T,
}

impl<T: DSend + ToOwned> DBox<'_, T> {
    pub fn size(&self) -> usize {
        self._inner.size()
    }

    pub fn pass(&mut self) -> &mut DPtr<'_, ()> {
        self._inner.pass()
    }
}

impl<T: DeepCopy + ToOwned> DBox<'_, T> {
    pub fn consume(self) -> Result<T, CUDAError> {
        let size = self._inner._size;
        let data = self._host_data;
        unsafe { 
            sys::cuMemcpyDtoH_v2(&data as *const T as *mut c_void, self._inner._device_ptr as u64, size).to_result()?;
            Ok(data)
        }
    }

    pub fn copy_to_host(&self, container: &mut T) -> Result<(), CUDAError> {
        self._inner.copy_to_host(container)
    }
}

impl<'a, T: DeepCopy + ToOwned + Clone> DBox<'a, Vec<T>> {
    pub fn consume(self) -> Result<Vec<T>, CUDAError> {
        let mut data = self._host_data;
        self._inner.copy_to_host(&mut data)?;
        Ok(data)
    }

    pub fn as_mut_slice(&mut self) -> &mut DPtr<'a, &mut [T]> {
        unsafe { transmute(&mut self._inner) }
    }
}