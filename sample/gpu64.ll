; NVVM IR version 1
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%adt2 = type {i64, i32, [1 x i8], i32, [9 x i8], [9 x i8]}
%adt5 = type {}
%adt1 = type {[0 x i8**]*, [9 x i8], [0 x %adt3]*}
%adt4 = type {i32, i32, [1 x i8], [9 x i8], [9 x i8], i8**}
%adt0 = type {i8**, [9 x i8], %adt6*, i1, i1}
%adt3 = type {[17 x i8]}
%adt6 = type {i8**, i32, i32}
@static_0 = constant {[8 x i8], void} {[8 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0], undef}
@static_1 = constant bitcast {[8 x i8], void} {[8 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0], undef} to i8* i8*
@static_2 = constant getelementptr i8*, {[8 x i8], void} {[8 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0], undef}, [0] i8*
@static_3 = constant {{[12 x i8]}, [8 x i8]} {{[12 x i8]} {[12 x i8] [i8 105, i8 110, i8 118, i8 97, i8 108, i8 105, i8 100, i8 32, i8 97, i8 114, i8 103, i8 115]}, [8 x i8] [i8 12, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0]}
@static_4 = constant {{[31 x i8]}, [8 x i8]} {{[31 x i8]} {[31 x i8] [i8 84, i8 104, i8 105, i8 115, i8 32, i8 105, i8 115, i8 32, i8 97, i8 32, i8 112, i8 97, i8 110, i8 105, i8 99, i8 32, i8 102, i8 114, i8 111, i8 109, i8 32, i8 116, i8 104, i8 101, i8 32, i8 107, i8 101, i8 114, i8 110, i8 101, i8 108]}, [8 x i8] [i8 31, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0]}

define void @_ZN12rust_kernels5gpu6417h5ad03dc6d7471ae6E(i8* %a, i64 %b) {
start:
  %0 = alloca i8, i64 48, align 8
  %1 = alloca i8, i64 16, align 8
  %2 = bitcast i8* %1 to i8**
  store i8* %a, i8** %2, align 8
  %3 = getelementptr inbounds i8, i8* %1,i64 8
  %4 = bitcast i8* %3 to i64*
  store i64 %b, i64* %4, align 8
  %5 = bitcast i8* %0 to {{i8*, i64}, {i8*, i64}, {i8*, i64}}*
  %6 = bitcast {{[31 x i8]}, [8 x i8]}* @static_4 to [0 x i8**]*
  call void @_ZN4core3fmt9Arguments9new_const17h062b9258b982828aE({{i8*, i64}, {i8*, i64}, {i8*, i64}}* %5, [0 x i8**]* %6, i64 1)
  %7 = bitcast i8* %0 to %adt0**
  %8 = load %adt0*, %adt0** %7, align 8
  call void @kernel_panic_fmt_impl(%adt0* %8)
  unreachable
}


define void @kernel_panic_fmt_impl(%adt0* %a) {
start:
  %0 = alloca i8, i64 8, align 8
  %1 = bitcast i8* %0 to %adt0**
  store %adt0* %a, %adt0** %1, align 8
  call void @__trap()
  br label %bb1
bb1:
  br label %bb1
}


define void @_ZN4core3fmt9Arguments9new_const17h062b9258b982828aE({{i8*, i64}, {i8*, i64}, {i8*, i64}}* %a, i8* %b, i64 %c) {
start:
  %0 = alloca i8, i64 48, align 8
  %1 = alloca i8, i64 16, align 8
  %2 = bitcast i8* %1 to i8**
  store i8* %b, i8** %2, align 8
  %3 = getelementptr inbounds i8, i8* %1,i64 8
  %4 = bitcast i8* %3 to i64*
  store i64 %c, i64* %4, align 8
  %5 = icmp ugt i64 %c, 1
  br i1 %5, label %bb1, label %bb3
bb3:
  %6 = bitcast {{i8*, i64}, {i8*, i64}, {i8*, i64}}* %a to i8**
  store i8* %b, i8** %6, align 8
  %7 = bitcast {{i8*, i64}, {i8*, i64}, {i8*, i64}}* %a to i8*
  %8 = getelementptr inbounds i8, i8* %7,i64 8
  %9 = bitcast i8* %8 to i64*
  store i64 %c, i64* %9, align 8
  %10 = load i8*, i8** @static_2, align 8
  %11 = bitcast i8** @static_2 to i8*
  %12 = getelementptr inbounds i8, i8* %11,i64 8
  %13 = bitcast i8* %12 to i64*
  %14 = load i64, i64* %13, align 8
  %15 = bitcast {{i8*, i64}, {i8*, i64}, {i8*, i64}}* %a to i8*
  %16 = getelementptr inbounds i8, i8* %15,i64 32
  %17 = bitcast i8* %16 to i8**
  store i8* %10, i8** %17, align 8
  %18 = getelementptr inbounds i8, i8* %16,i64 8
  %19 = bitcast i8* %18 to i64*
  store i64 %14, i64* %19, align 8
  %20 = bitcast {{i8*, i64}, {i8*, i64}, {i8*, i64}}* %a to i8*
  %21 = getelementptr inbounds i8, i8* %20,i64 16
  %22 = bitcast i8* %21 to {}*
  store {} {}, {}* %22, align 8
  %23 = getelementptr inbounds i8, i8* %21,i64 8
  %24 = bitcast i8* %23 to i64*
  store i64 0, i64* %24, align 8
  ret void
bb1:
  %25 = bitcast i8* %0 to {{i8*, i64}, {i8*, i64}, {i8*, i64}}*
  %26 = bitcast {{[12 x i8]}, [8 x i8]}* @static_3 to [0 x i8**]*
  call void @_ZN4core3fmt9Arguments9new_const17h062b9258b982828aE({{i8*, i64}, {i8*, i64}, {i8*, i64}}* %25, [0 x i8**]* %26, i64 1)
  %27 = bitcast i8* %0 to %adt0**
  %28 = load %adt0*, %adt0** %27, align 8
  call void @kernel_panic_fmt_impl(%adt0* %28)
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
!1 = !{void(i8*, i64)* @_ZN12rust_kernels5gpu6417h5ad03dc6d7471ae6E, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0}

