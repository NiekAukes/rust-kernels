; NVVM IR version 1
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%adt0 = type {i32*, i64}
%adt1 = type {i8**, i32, i32}
@global0 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 50, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 235, i8 1, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]
@global1 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 19, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]
@global2 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 19, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]
@global3 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 20, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]
@global4 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 20, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]
@global5 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 27, i8 0, i8 0, i8 0, i8 20, i8 0, i8 0, i8 0]
@global6 = constant [24 x i8] [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 27, i8 0, i8 0, i8 0, i8 42, i8 0, i8 0, i8 0]

define i32 @_ZN4cuda3gpu12global_tid_x17ha36a33b0294aba95E() {
start:
  %0 = call i32 @__nvvm_thread_idx_x()
  %1 = alloca i8, i64 4, align 4
  %2 = bitcast i8* %1 to i32*
  store i32 %0, i32* %2, align 4
  %3 = call i32 @__nvvm_block_idx_x()
  %4 = alloca i8, i64 4, align 4
  %5 = bitcast i8* %4 to i32*
  store i32 %3, i32* %5, align 4
  %6 = call i32 @__nvvm_block_dim_x()
  %7 = alloca i8, i64 4, align 4
  %8 = bitcast i8* %7 to i32*
  store i32 %6, i32* %8, align 4
  %9 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %3, i32 %6)
  %10 = extractvalue {i32, i1} %9, 0
  %11 = extractvalue {i32, i1} %9, 1
  %12 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %0, i32 %10)
  %13 = extractvalue {i32, i1} %12, 0
  %14 = extractvalue {i32, i1} %12, 1
  ret i32 %13
}


define void @_ZN12rust_kernels10matrix_mul17h6bf2fc4dc43165baE(i8* %a, i64 %b, i8* %c, i64 %d, i8* %e, i64 %f, i32 %g) {
start:
  %0 = alloca i8, i64 16, align 8
  %1 = bitcast i8* %0 to i8**
  store i8* %e, i8** %1, align 8
  %2 = getelementptr inbounds i8, i8* %0,i64 8
  %3 = bitcast i8* %2 to i64*
  store i64 %f, i64* %3, align 8
  %4 = alloca i8, i64 4, align 4
  %5 = alloca i8, i64 4, align 4
  %6 = alloca i8, i64 16, align 8
  %7 = bitcast i8* %6 to i8**
  store i8* %a, i8** %7, align 8
  %8 = getelementptr inbounds i8, i8* %6,i64 8
  %9 = bitcast i8* %8 to i64*
  store i64 %b, i64* %9, align 8
  %10 = alloca i8, i64 16, align 8
  %11 = bitcast i8* %10 to i8**
  store i8* %c, i8** %11, align 8
  %12 = getelementptr inbounds i8, i8* %10,i64 8
  %13 = bitcast i8* %12 to i64*
  store i64 %d, i64* %13, align 8
  %14 = alloca i8, i64 4, align 4
  %15 = bitcast i8* %14 to i32*
  store i32 %g, i32* %15, align 4
  %16 = call i32 @_ZN4cuda3gpu12global_tid_x17ha36a33b0294aba95E()
  %17 = alloca i8, i64 4, align 4
  %18 = bitcast i8* %17 to i32*
  store i32 %16, i32* %18, align 4
  %19 = icmp eq i32 %g, 0
  br i1 %19, label %panic, label %bb2
bb2:
  %20 = icmp eq i32 %g, -1
  %21 = icmp eq i32 %16, -2147483648
  %22 = and i1 %20, %21
  br i1 %22, label %panic, label %bb3
bb3:
  %23 = sdiv i32 %16, %g
  %24 = alloca i8, i64 4, align 4
  %25 = bitcast i8* %24 to i32*
  store i32 %23, i32* %25, align 4
  %26 = icmp eq i32 %g, 0
  br i1 %26, label %panic, label %bb4
bb4:
  %27 = icmp eq i32 %g, -1
  %28 = icmp eq i32 %16, -2147483648
  %29 = and i1 %27, %28
  br i1 %29, label %panic, label %bb5
bb5:
  %30 = srem i32 %16, %g
  %31 = alloca i8, i64 4, align 4
  %32 = bitcast i8* %31 to i32*
  store i32 %30, i32* %32, align 4
  %33 = icmp slt i32 %23, %g
  br i1 %33, label %bb6, label %bb22
bb22:
  ret void
bb6:
  %34 = icmp slt i32 %30, %g
  br i1 %34, label %bb7, label %bb22
bb7:
  %35 = bitcast i8* %4 to i32*
  store i32 0, i32* %35, align 4
  %36 = bitcast i8* %5 to i32*
  store i32 0, i32* %36, align 4
  br label %bb8
bb8:
  %37 = bitcast i8* %5 to i32*
  %38 = load i32, i32* %37, align 4
  %39 = icmp slt i32 %38, %g
  br i1 %39, label %bb9, label %bb19
bb19:
  %40 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %23, i32 %g)
  %41 = extractvalue {i32, i1} %40, 0
  %42 = extractvalue {i32, i1} %40, 1
  %43 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %41, i32 %30)
  %44 = extractvalue {i32, i1} %43, 0
  %45 = extractvalue {i32, i1} %43, 1
  %46 = alloca i8, i64 4, align 4
  %47 = bitcast i8* %46 to i32*
  store i32 %44, i32* %47, align 4
  %48 = sext i32 %44 to i64
  %49 = bitcast i8* %4 to i32*
  %50 = load i32, i32* %49, align 4
  %51 = bitcast i8* %0 to %adt0*
  call void @_ZN4cuda4dmem15Buffer$LT$T$GT$3set17h98f419e728313cf8E(%adt0* %51, i64 %48, i32 %50)
  br label %bb22
bb9:
  %52 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %23, i32 %g)
  %53 = extractvalue {i32, i1} %52, 0
  %54 = extractvalue {i32, i1} %52, 1
  %55 = bitcast i8* %5 to i32*
  %56 = load i32, i32* %55, align 4
  %57 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %53, i32 %56)
  %58 = extractvalue {i32, i1} %57, 0
  %59 = extractvalue {i32, i1} %57, 1
  %60 = alloca i8, i64 4, align 4
  %61 = bitcast i8* %60 to i32*
  store i32 %58, i32* %61, align 4
  %62 = bitcast i8* %5 to i32*
  %63 = load i32, i32* %62, align 4
  %64 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %63, i32 %g)
  %65 = extractvalue {i32, i1} %64, 0
  %66 = extractvalue {i32, i1} %64, 1
  %67 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %65, i32 %30)
  %68 = extractvalue {i32, i1} %67, 0
  %69 = extractvalue {i32, i1} %67, 1
  %70 = alloca i8, i64 4, align 4
  %71 = bitcast i8* %70 to i32*
  store i32 %68, i32* %71, align 4
  %72 = sext i32 %58 to i64
  %73 = icmp ult i64 %72, %b
  br i1 %73, label %bb14, label %panic
bb14:
  %74 = bitcast i8* %a to [0 x i32]*
  %75 = getelementptr inbounds [0 x i32], [0 x i32]* %74,i64 0,i64 %72
  %76 = load i32, i32* %75, align 4
  %77 = sext i32 %68 to i64
  %78 = icmp ult i64 %77, %d
  br i1 %78, label %bb15, label %panic
bb15:
  %79 = bitcast i8* %c to [0 x i32]*
  %80 = getelementptr inbounds [0 x i32], [0 x i32]* %79,i64 0,i64 %77
  %81 = load i32, i32* %80, align 4
  %82 = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %76, i32 %81)
  %83 = extractvalue {i32, i1} %82, 0
  %84 = extractvalue {i32, i1} %82, 1
  %85 = bitcast i8* %4 to i32*
  %86 = load i32, i32* %85, align 4
  %87 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %86, i32 %83)
  %88 = extractvalue {i32, i1} %87, 0
  %89 = extractvalue {i32, i1} %87, 1
  %90 = bitcast i8* %4 to i32*
  store i32 %88, i32* %90, align 4
  %91 = bitcast i8* %5 to i32*
  %92 = load i32, i32* %91, align 4
  %93 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %92, i32 1)
  %94 = extractvalue {i32, i1} %93, 0
  %95 = extractvalue {i32, i1} %93, 1
  %96 = bitcast i8* %5 to i32*
  store i32 %94, i32* %96, align 4
  br label %bb8
panic:
  call void @llvm.trap()
  unreachable
}


define i32* @_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add17h2f83b7147c177dd8E(i32* %a, i64 %b) {
start:
  %0 = alloca i8, i64 8, align 8
  %1 = bitcast i8* %0 to i32**
  store i32* %a, i32** %1, align 8
  %2 = alloca i8, i64 8, align 8
  %3 = bitcast i8* %2 to i64*
  store i64 %b, i64* %3, align 8
  %4 = getelementptr inbounds i32, i32* %a,i64 %b
  ret i32* %4
}


define void @_ZN4cuda4dmem15Buffer$LT$T$GT$3set17h98f419e728313cf8E(%adt0* %a, i64 %b, i32 %c) {
start:
  %0 = alloca i8, i64 8, align 8
  %1 = bitcast i8* %0 to %adt0**
  store %adt0* %a, %adt0** %1, align 8
  %2 = alloca i8, i64 8, align 8
  %3 = bitcast i8* %2 to i64*
  store i64 %b, i64* %3, align 8
  %4 = alloca i8, i64 4, align 4
  %5 = bitcast i8* %4 to i32*
  store i32 %c, i32* %5, align 4
  %6 = bitcast %adt0* %a to i8*
  %7 = getelementptr inbounds i8, i8* %6,i64 8
  %8 = bitcast i8* %7 to i64*
  %9 = load i64, i64* %8, align 8
  %10 = icmp uge i64 %b, %9
  br i1 %10, label %bb1, label %bb2
bb2:
  %11 = bitcast %adt0* %a to i32**
  %12 = load i32*, i32** %11, align 8
  %13 = call i32* @_ZN4core3ptr7mut_ptr31_$LT$impl$u20$$BP$mut$u20$T$GT$3add17h2f83b7147c177dd8E(i32* %12, i64 %b)
  %14 = alloca i8, i64 8, align 8
  %15 = bitcast i8* %14 to i32**
  store i32* %13, i32** %15, align 8
  %16 = bitcast i32* %13 to i8*
  %17 = ptrtoint i8* %16 to i64
  %18 = sub i64 4, 1
  %19 = and i64 %17, %18
  %20 = icmp eq i64 %19, 0
  br i1 %20, label %bb4, label %panic
bb1:
  call void @__trap()
  br label %bb2
bb4:
  store i32 %c, i32* %13, align 4
  ret void
panic:
  call void @llvm.trap()
  unreachable
}


declare void @llvm.trap()
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)



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
!1 = !{void(i8*, i64, i8*, i64, i8*, i64, i32)* @_ZN12rust_kernels10matrix_mul17h6bf2fc4dc43165baE, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0}

