use std::{
    arch::asm,
    ffi::c_void,
    mem::{transmute, MaybeUninit},
    ops::{Deref, DerefMut},
};

use crate::{
    dmem::{DPassMode, DPtr, DSend},
    sys, CUDAError, ToResult,
};

/// an atomic integer specifically for gpu usage
#[derive(Debug, Clone, Copy)]
pub struct AtomI32 {
    value: i32,
}

impl AtomI32 {
    pub fn new(value: i32) -> Self {
        AtomI32 { value }
    }

    // #[target = "nvvm"]
    // pub fn load(&self) -> i32 {
    //     unsafe {
    //         let mut out: i32;
    //         asm!(
    //             "ld.global.s32 {out}, [{ptr}];",
    //             ptr = in(reg) &self.value as *const i32 as u64,
    //             out = lateout(reg) out,
    //             options(nostack, preserves_flags)
    //         );
    //         out
    //     }
    // }

    // #[target = "nvvm"]
    // pub fn store(&mut self, value: i32) {
    //     unsafe {
    //         asm!(
    //             "st.global.s32 [{ptr}], {val};",
    //             ptr = in(reg) &mut self.value as *mut i32 as u64,
    //             val = in(reg) value,
    //             options(nostack, preserves_flags)
    //         );
    //     }
    // }

    // #[target = "nvvm"]
    // pub fn add(&mut self, value: i32) -> i32 {
    //     let mut old: i32;
    //     unsafe {
    //         //atomic add
    //         asm!(
    //             "atom.global.add.s32 {old}, [{ptr}], {val};",
    //             ptr = in(reg) &mut self.value as *mut i32 as u64,
    //             val = in(reg) value,
    //             old = lateout(reg) old,
    //             options(nostack, preserves_flags)
    //         );
    //         old
    //     }
    // }

    // #[target = "nvvm"]
    // pub fn sub(&mut self, value: i32) -> i32 {
    //     let mut old: i32;
    //     // subtract by adding a negative value
    //     unsafe {
    //         asm!(
    //             "atom.global.add.s32 {old}, [{ptr}], {neg};",
    //             ptr = in(reg) &mut self.value as *mut i32 as u64,
    //             neg = in(reg) -value,
    //             old = lateout(reg) old,
    //             options(nostack, preserves_flags)
    //         );
    //         old
    //     }
    // }
}

#[target = "nvvm"]
#[inline(always)]
pub fn atom_add_shared(obj: &mut AtomI32, value: i32) -> i32 {
    let mut old: i32;
    unsafe {
        asm!(
            "atom.global.add.s32 {old}, [{ptr}], {val};",
            ptr = in(reg) &mut obj.value as *mut i32 as u64,
            val = in(reg) value,
            old = lateout(reg) old,
            options(nostack, preserves_flags)
        );

        old
    }
}

impl Deref for AtomI32 {
    type Target = i32;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl DerefMut for AtomI32 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut i32 {
        &mut self.value
    }
}

/// Shared object, this object is both accesible from the host and the device.
#[derive(Debug, Clone, Copy)]
pub struct Shared<T: Sized + Copy> {
    primary: *mut T, // the primary pointer is always on the current device
    other: *mut T,
}

impl<T: Sized + Copy> Shared<T> {
    /// Allocates a new T on the GPU and returns a Shared object.
    /// This function should be called from the host.
    pub fn alloc() -> Result<Self, CUDAError> {
        crate::get_cuda(); // ensure CUDA is initialized
        unsafe {
            let mut host_ptr = MaybeUninit::uninit();
            let mut device_ptr = MaybeUninit::uninit();
            sys::cuMemAllocHost_v2(host_ptr.as_mut_ptr(), std::mem::size_of::<T>()).to_result()?;
            let h = host_ptr.assume_init() as *mut T;
            sys::cuMemHostGetDevicePointer_v2(
                device_ptr.as_mut_ptr(),
                host_ptr.as_mut_ptr() as *mut c_void,
                0,
            );
            let d = device_ptr.assume_init() as *mut T;

            Ok(Self {
                primary: h,
                other: d,
            })
        }
    }

    pub fn new(value: T) -> Result<Self, CUDAError> {
        let shared = Self::alloc()?;
        unsafe {
            *shared.primary = value; // initialize the primary pointer
            *shared.other = value; // initialize the other pointer
        }
        Ok(shared)
    }
}

impl<T: Sized + Copy> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.primary }
    }
}

impl<T: Sized + Copy + DerefMut> DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.primary }
    }
}

impl<T: Copy + DSend> DSend for Shared<T> {
    fn to_device(&self) -> Result<DPtr<Self>, CUDAError> {
        let cuda = crate::get_cuda();
        let size = std::mem::size_of::<T>();
        Ok(DPtr {
            cuda,
            // the primary pointer is switched so that it is always on the current device
            _device_ptr: unsafe { transmute(self.other) },
            _size: size,
            _pass_mode: DPassMode::Pair {
                data: unsafe { transmute(self.primary) },
                _size: size,
            },
        })
    }

    fn copy_from_device(&mut self, dptr: DPtr<Self>) -> Result<(), CUDAError> {
        Ok(())
    }
}

impl<T: Copy + DSend> DPtr<Shared<T>> {
    pub fn retrieve(&self) -> Result<Shared<T>, CUDAError> {
        // reconstruct the shared memory
        let primary = unsafe { transmute(self._device_ptr) };
        let other = unsafe {
            match self._pass_mode {
                DPassMode::Pair { data, _size } => transmute(data),
                _ => panic!("Shared memory must be passed as a pair"),
            }
        };
        Ok(Shared { primary, other })
    }
}
