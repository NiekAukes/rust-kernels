use std::{
    mem::{transmute, MaybeUninit},
    os::raw::c_void,
};

use super::CUDA;
use crate::{gpu::__trap, sys, CUDAError, ToResult};

// ==================
// ===== Traits =====
// ==================

/// A trait that denotes that this type does not have any references to host memory
/// and can be safely copied to an external device without
pub trait DeepCopy {}
// impl DeepCopy for u8 {}
// impl DeepCopy for u16 {}
// impl DeepCopy for u32 {}
// impl DeepCopy for u64 {}
// impl DeepCopy for i8 {}
// impl DeepCopy for i16 {}
// impl DeepCopy for i32 {}
// impl DeepCopy for i64 {}
// impl DeepCopy for f32 {}
// impl DeepCopy for f64 {}
// impl DeepCopy for bool {}
// impl DeepCopy for char {}
// impl DeepCopy for () {}

#[macro_export]
macro_rules! deepcopy {
    ($t:ty) => {
        impl DeepCopy for $t {}
        impl DSend for $t {
            fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
                let cuda = crate::get_cuda();
                let size = std::mem::size_of::<$t>();
                let data: &u64 = unsafe { transmute(self) };
                let data = *data;
                Ok(DPtr {
                    cuda,
                    _device_ptr: 0 as *mut Self,
                    _size: size,
                    _pass_mode: DPassMode::Scalar { data },
                })
            }

            fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
                Ok(())
            }
        }
        impl DSend for &$t {
            fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
                let cuda = crate::get_cuda();
                let size = std::mem::size_of::<$t>();
                let mut dptr = MaybeUninit::uninit();
                let device_ptr = unsafe {
                    sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
                    dptr.assume_init() as *mut Self
                };
                unsafe {
                    sys::cuMemcpyHtoD_v2(
                        device_ptr as u64,
                        *self as *const $t as *const c_void,
                        size,
                    )
                    .to_result()?;
                }
                Ok(DPtr {
                    cuda,
                    _device_ptr: device_ptr,
                    _size: size,
                    _pass_mode: DPassMode::Direct,
                })
            }

            fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
                Ok(())
            }
        }

        impl DPtr<'_, $t> {
            pub fn copy_to_host(&self, container: &mut $t) -> Result<(), CUDAError> {
                let size = self._size;
                let data = container as *mut $t as *mut c_void;
                unsafe {
                    sys::cuMemcpyDtoH_v2(data, self._device_ptr as u64, size).to_result()?;
                    Ok(())
                }
            }
        }

        impl DBox<'_, $t> {
            pub fn consume(self) -> Result<$t, CUDAError> {
                let size = self._inner._size;
                let data = self._host_data;
                unsafe {
                    sys::cuMemcpyDtoH_v2(
                        &data as *const $t as *mut c_void,
                        self._inner._device_ptr as u64,
                        size,
                    )
                    .to_result()?;
                    Ok(data)
                }
            }

            pub fn copy_to_host(&self, container: &mut $t) -> Result<(), CUDAError> {
                self._inner.copy_to_host(container)
            }
        }
    };
}

impl DeepCopy for () {}
impl DSend for () {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
        panic!("Cannot copy a unit type to device")
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        panic!("Cannot copy a unit type from device")
    }
}

impl DPtr<'_, ()> {
    pub fn copy_to_host(&self, _container: &mut ()) -> Result<(), CUDAError> {
        Ok(())
    }
}

deepcopy!(u8);
deepcopy!(u16);
deepcopy!(u32);
deepcopy!(u64);
deepcopy!(usize);
deepcopy!(i8);
deepcopy!(i16);
deepcopy!(i32);
deepcopy!(i64);
deepcopy!(f32);
deepcopy!(f64);
deepcopy!(bool);
deepcopy!(char);


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

#[derive(Debug)]
pub(crate) enum DPassMode {
    Direct,
    Scalar { data: u64 },
    Pair { data: u64, _size: usize },

}

/// A reference to data on the device
pub struct DPtr<'a, T: DSend> {
    pub(crate) cuda: &'a CUDA,
    pub(crate) _device_ptr: *mut T,
    pub(crate) _size: usize,
    pub(crate) _pass_mode: DPassMode,
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

// impl<'a, T: DeepCopy> DPtr<'a, T> {
//     pub fn copy_to_host(&self, container: &mut T) -> Result<(), CUDAError> {
//         let size = self._size;
//         let data = container as *mut T as *mut c_void;
//         unsafe {
//             sys::cuMemcpyDtoH_v2(data, self._device_ptr as u64, size).to_result()?;
//             Ok(())
//         }
//     }
// }

// impl<T: DeepCopy> DSend for T {
//     fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
//         let cuda = crate::get_cuda();
//         let size = std::mem::size_of::<T>();
//         let mut dptr = MaybeUninit::uninit();
//         let device_ptr = unsafe {
//             sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
//             dptr.assume_init() as *mut Self
//         };
//         unsafe {
//             sys::cuMemcpyHtoD_v2(device_ptr as u64, self as *const T as *const c_void, size)
//                 .to_result()?;
//         }
//         Ok(DPtr {
//             cuda,
//             _device_ptr: device_ptr,
//             _size: size,
//             _pass_mode: DPassMode::Direct,
//         })
//     }

//     fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
//         //dptr.copy_to_host(self)
//         Ok(())
//     }
// }

// impl<T: DeepCopy> DSend for &T {
//     fn to_device<'a>(self) -> Result<DPtr<'a, Self>, CUDAError> {
//         let cuda = crate::get_cuda();
//         let size = std::mem::size_of::<T>();
//         let mut dptr = MaybeUninit::uninit();
//         let device_ptr = unsafe {
//             sys::cuMemAlloc_v2(dptr.as_mut_ptr(), size).to_result()?;
//             dptr.assume_init() as *mut Self
//         };
//         unsafe {
//             sys::cuMemcpyHtoD_v2(device_ptr as u64, self as *const T as *const c_void, size)
//                 .to_result()?;
//         }
//         Ok(DPtr {
//             cuda,
//             _device_ptr: device_ptr,
//             _size: size,
//             _pass_mode: DPassMode::Direct,
//         })
//     }

//     fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
//         //dptr.copy_to_host(self)
//         Ok(())
//     }
// }

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
            _pass_mode: DPassMode::Pair {
                data: self.len() as u64,
                _size: 8,
            },
        })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        // we can't copy to an immutable slice, so just do nothing
        //Err(CUDAError::Unknown(0)) // should not even be possible
        Ok(())
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
            _pass_mode: DPassMode::Pair {
                data: self.len() as u64,
                _size: 8,
            },
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
            _pass_mode: DPassMode::Direct, // TODO: check this
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

fn consume_box<T: DSend + ToOwned>(dbox: DBox<T>) -> Result<T, CUDAError> {
    let size = dbox.size();
    let data = dbox._host_data;
    unsafe {
        sys::cuMemcpyDtoH_v2(
            &data as *const T as *mut c_void,
            dbox._inner._device_ptr as u64,
            size,
        )
        .to_result()?;
        Ok(data)
    }
}

// impl<T: DeepCopy + ToOwned + DSend> DBox<'_, T> {
//     pub fn consume(self) -> Result<T, CUDAError> {
//         let size = self._inner._size;
//         let data = self._host_data;
//         unsafe {
//             sys::cuMemcpyDtoH_v2(
//                 &data as *const T as *mut c_void,
//                 self._inner._device_ptr as u64,
//                 size,
//             )
//             .to_result()?;
//             Ok(data)
//         }
//     }

//     pub fn copy_to_host(&self, container: &mut T) -> Result<(), CUDAError> {
//         self._inner.copy_to_host(container)
//     }
// }

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

impl<'a, 'b, T: DeepCopy + ToOwned + Clone> DPtr<'a, &'b [T]> {
    pub fn to_raw_ptr(&self) -> DPtr<'a, &'b [T]> {
        DPtr {
            cuda: self.cuda,
            _device_ptr: self._device_ptr,
            _size: self._size,
            _pass_mode: DPassMode::Direct,
        }
    }
}

// pub fn new_raw_ptr<'a>(ptr: u64, cuda: &'a CUDA) -> DPtr<'a, ()> {
//     DPtr {
//         cuda,
//         _device_ptr: ptr as *mut (),
//         _size: 8,
//         _pass_mode: DPassMode::Direct,
//     }
// }

#[derive(Debug, Clone, Copy)]
pub struct Buffer<T: Sized + Copy>(*mut T, usize);

impl<T: Sized + Copy> Buffer<T> {
    pub fn get(&self, index: usize) -> T {
        if index >= self.1 {
            //panic!("Index out of bounds");
            unsafe {
                __trap();
            }
        }
        unsafe { *self.0.add(index) }
    }

    pub fn set(&mut self, index: usize, value: T) {
        if index >= self.1 {
            //panic!("Index out of bounds");
            unsafe {
                __trap();
            }
        }
        unsafe {
            let ptr = self.0.add(index);
            *ptr = value;
        }
    }

    pub unsafe fn get_uc(&self, index: usize) -> T {
        *self.0.add(index)
    }

    pub unsafe fn set_uc(&mut self, index: usize, value: T) {
        let ptr = self.0.add(index);
        *ptr = value;
    }

    /// Allocates a modifyable buffer of the given length on the device
    pub fn alloc(size: usize) -> Result<Self, CUDAError> {
        let cuda = crate::get_cuda();
        let mut dptr = MaybeUninit::uninit();
        // get the size of the type
        let bc = size * std::mem::size_of::<T>();
        // allocate the device memory
        let device_ptr = unsafe {
            sys::cuMemAlloc_v2(dptr.as_mut_ptr(), bc).to_result()?;
            dptr.assume_init() as *mut T
        };
        Ok(Buffer(device_ptr, size))
    }

    pub fn retrieve(&self) -> Result<Vec<T>, CUDAError> {
        let mut data = Vec::with_capacity(self.1);
        let bc = self.1 * std::mem::size_of::<T>();
        unsafe {
            data.set_len(self.1);
        }
        unsafe {
            sys::cuMemcpyDtoH_v2(data.as_mut_ptr() as *mut c_void, self.0 as u64, bc)
                .to_result()?;
        }
        Ok(data)
    }

    fn free(&mut self) {
        unsafe {
            sys::cuMemFree_v2(self.0 as u64).to_result().unwrap();
        }
    }
}

impl<T: Sized + Copy> DSend for Buffer<T> {
    fn to_device<'a>(&self) -> Result<DPtr<'a, Self>, CUDAError> {
        let cuda = crate::get_cuda();
        let size = std::mem::size_of::<T>() * self.1;
        Ok(DPtr {
            cuda,
            _device_ptr: unsafe { transmute(self.0) },
            _size: size,
            _pass_mode: DPassMode::Pair {
                data: self.1 as u64,
                _size: 8,
            },
        })
    }

    fn copy_from_device<'a>(&mut self, dptr: DPtr<'a, Self>) -> Result<(), CUDAError> {
        Ok(())
    }
}

impl<T: Sized + Copy> DPtr<'_, Buffer<T>> {
    pub fn retrieve(&self) -> Result<Vec<T>, CUDAError> {
        let size = match self._pass_mode {
            DPassMode::Pair { data, _size } => data as usize,
            _ => panic!("unexpected pass mode"),
        };
        let mut data = Vec::with_capacity(size);
        let bc = size * std::mem::size_of::<T>();
        unsafe {
            data.set_len(size);
        }
        unsafe {
            sys::cuMemcpyDtoH_v2(
                data.as_mut_ptr() as *mut c_void,
                self._device_ptr as u64,
                bc,
            )
            .to_result()?;
        }
        Ok(data)
    }
}

// =================================================================
// Atomic i32
// =================================================================

// #[derive(Debug, Clone, Copy)]
// struct Ai32 {
//     x: i32,
// }

// impl Ai32 {
//     pub fn alloc(&self) -> i32 {
//         self.x
//     }

//     pub fn new(x: i32) -> Self {
//         Ai32 { x }
//     }

//     pub fn set(&mut self, x: i32) {
//         self.x = x;
//     }
// }



deepcopy!(crate::atom::AtomI32);