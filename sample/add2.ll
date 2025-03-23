; NVVM IR version 1
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%adt1 = type {i8**, i32, i32}
%adt0 = type {i32*, i64}
@global0 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 23, i8 0, i8 0, i8 0, i8 16, i8 0, i8 0, i8 0]
@global1 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 23, i8 0, i8 0, i8 0, i8 23, i8 0, i8 0, i8 0]

define void @_ZN4cuda4dmem15Buffer$LT$T$GT$3set17hd26f74f4e653e083E(%adt0* %a, i64 %b, i32 %c) {
start:
  %0 = bitcast %adt0* %a to i8*
  %1 = getelementptr inbounds i8, i8* %0,i64 8
  %2 = bitcast i8* %1 to i64*
  %3 = load i64, i64* %2, align 8
  %4 = icmp uge i64 %b, %3
  br i1 %4, label %bb1, label %bb2
bb2:
  br label %bb3
bb1:
  call void @__trap()
  br label %bb3
bb3:
  %5 = bitcast %adt0* %a to i32**
  %6 = load i32*, i32** %5, align 8
  %7 = getelementptr inbounds i32, i32* %6,i64 %b
  store i32 %c, i32* %7, align 4
  ret void
}


define void @_ZN12rust_kernels4add217hcefc1cbdca976471E(i8* %a, i64 %b, i8* %c, i64 %d, i8* %e, i64 %f) {
start:
  %0 = alloca i8, i64 16
  %1 = bitcast i8* %0 to i8**
  store i8* %e, i8** %1, align 8
  %2 = getelementptr inbounds i8, i8* %0,i64 8
  %3 = bitcast i8* %2 to i64*
  store i64 %f, i64* %3, align 8
  %4 = call i32 @__nvvm_thread_idx_x()
  %5 = call i32 @__nvvm_block_idx_x()
  %6 = call i32 @__nvvm_block_dim_x()
  %7 = mul i32 %5, %6
  %8 = add i32 %4, %7
  %9 = sext i32 %8 to i64
  %10 = icmp ult i64 %9, %b
  br i1 %10, label %bb1, label %panic
bb1:
  %11 = bitcast i8* %a to [0 x i32]*
  %12 = getelementptr inbounds [0 x i32], [0 x i32]* %11,i64 0,i64 %9
  %13 = load i32, i32* %12, align 4
  %14 = icmp ult i64 %9, %d
  br i1 %14, label %bb2, label %panic
bb2:
  %15 = bitcast i8* %c to [0 x i32]*
  %16 = getelementptr inbounds [0 x i32], [0 x i32]* %15,i64 0,i64 %9
  %17 = load i32, i32* %16, align 4
  %18 = add i32 %13, %17
  %19 = bitcast i8* %0 to %adt0*
  call void @_ZN4cuda4dmem15Buffer$LT$T$GT$3set17hd26f74f4e653e083E(%adt0* %19, i64 %9, i32 %18)
  ret void
panic:
  call void @llvm.trap()
  unreachable
}


declare void @llvm.trap()



; This is a hand-written llvm ir module which contains extra functions
; that are easier to write. They mostly contain nvvm intrinsics that are wrapped in new 
; functions so that rustc does not think they are llvm intrinsics and so you don't need to always use nightly for that.
;
; if you update this make sure to update libintrinsics.bc by running llvm-as (make sure you are using llvm-7 or it won't work when
; loaded into libnvvm).

define linkonce i32 @__nvvm_thread_idx_x() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  ret i32 %0
}

define linkonce i32 @__nvvm_thread_idx_y() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  ret i32 %0
}

define linkonce i32 @__nvvm_thread_idx_z() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  ret i32 %0
}

define linkonce i32 @__nvvm_block_idx_x() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  ret i32 %0
}

define linkonce i32 @__nvvm_block_idx_y() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  ret i32 %0
}

define linkonce i32 @__nvvm_block_idx_z() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  ret i32 %0
}

define linkonce i32 @__nvvm_block_dim_x() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  ret i32 %0
}

define linkonce i32 @__nvvm_block_dim_y() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  ret i32 %0
}

define linkonce i32 @__nvvm_block_dim_z() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  ret i32 %0
}

define linkonce i32 @__nvvm_grid_dim_x() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  ret i32 %0
}

define linkonce i32 @__nvvm_grid_dim_y() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  ret i32 %0
}

define linkonce i32 @__nvvm_grid_dim_z() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  ret i32 %0
}

define linkonce i32 @__nvvm_warp_size() alwaysinline {
start:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  ret i32 %0
}

define linkonce void @__trap() alwaysinline {
  start:
  call void @llvm.trap()
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()!nvvm.annotations = !{!1}
!1 = !{void(i8*, i64, i8*, i64, i8*, i64)* @_ZN12rust_kernels4add217hcefc1cbdca976471E, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0}

