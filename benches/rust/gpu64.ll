; NVVM IR version 1
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%adt0 = type {i32*, i64}
%adt1 = type {i8**, i32, i32}
@static_0 = constant {[4 x i8], [4 x i8]} {[4 x i8] [i8 0, i8 0, i8 0, i8 0], [4 x i8] undef}
@static_1 = constant i8* getelementptr (i8, i8* bitcast ({[4 x i8], [4 x i8]}* @static_0 to i8*), i64 0)
@static_2 = constant {[82 x i8]} {[82 x i8] [i8 117, i8 110, i8 115, i8 97, i8 102, i8 101, i8 32, i8 112, i8 114, i8 101, i8 99, i8 111, i8 110, i8 100, i8 105, i8 116, i8 105, i8 111, i8 110, i8 40, i8 115, i8 41, i8 32, i8 118, i8 105, i8 111, i8 108, i8 97, i8 116, i8 101, i8 100, i8 58, i8 32, i8 104, i8 105, i8 110, i8 116, i8 58, i8 58, i8 117, i8 110, i8 114, i8 101, i8 97, i8 99, i8 104, i8 97, i8 98, i8 108, i8 101, i8 95, i8 117, i8 110, i8 99, i8 104, i8 101, i8 99, i8 107, i8 101, i8 100, i8 32, i8 109, i8 117, i8 115, i8 116, i8 32, i8 110, i8 101, i8 118, i8 101, i8 114, i8 32, i8 98, i8 101, i8 32, i8 114, i8 101, i8 97, i8 99, i8 104, i8 101, i8 100]}
@static_3 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 22, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]}
@static_4 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 22, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]}
@static_5 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 23, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]}
@static_6 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 23, i8 0, i8 0, i8 0, i8 13, i8 0, i8 0, i8 0]}
@static_7 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 47, i8 0, i8 0, i8 0, i8 16, i8 0, i8 0, i8 0]}
@static_8 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 47, i8 0, i8 0, i8 0, i8 38, i8 0, i8 0, i8 0]}
@static_9 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 34, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_10 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 34, i8 0, i8 0, i8 0, i8 37, i8 0, i8 0, i8 0]}
@static_11 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 35, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_12 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 35, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}
@static_13 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 36, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_14 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 36, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}
@static_15 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 37, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_16 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 37, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}
@static_17 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 38, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_18 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 38, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}
@static_19 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 39, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_20 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 39, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}
@static_21 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 40, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_22 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 40, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}
@static_23 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 41, i8 0, i8 0, i8 0, i8 15, i8 0, i8 0, i8 0]}
@static_24 = constant {{[11 x i8]}, [16 x i8]} {{[11 x i8]} {[11 x i8] [i8 115, i8 114, i8 99, i8 47, i8 109, i8 97, i8 105, i8 110, i8 46, i8 114, i8 115]}, [16 x i8] [i8 11, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 41, i8 0, i8 0, i8 0, i8 43, i8 0, i8 0, i8 0]}

define void @_ZN4cuda3gpu26kernel_panic_nounwind_impl17h475d8c91e113d06eE(i8* %a, i64 %b) {
start:
  call void @__trap()
  br label %bb1
bb1:
  br label %bb1
}


define void @_ZN4cuda4dmem15Buffer$LT$T$GT$3set17h0322ab43b2f9fd10E(%adt0* %a, i64 %b, i32 %c) {
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


define void @_ZN4core4hint21unreachable_unchecked18precondition_check17h18af1954815f4ee4E() {
start:
  %0 = bitcast {[82 x i8]}* @static_2 to i8*
  call void @_ZN4cuda3gpu26kernel_panic_nounwind_impl17h475d8c91e113d06eE(i8* %0, i64 82)
  unreachable
}


define void @_ZN12rust_kernels10matrix_mul17h4f87c4a65a00d4eeE(i8* %a, i64 %b, i8* %c, i64 %d, i8* %e, i64 %f, i32 %g) {
start:
  %0 = alloca i8, i64 16, align 8
  %1 = bitcast i8* %0 to i8**
  store i8* %e, i8** %1, align 8
  %2 = getelementptr inbounds i8, i8* %0,i64 8
  %3 = bitcast i8* %2 to i64*
  store i64 %f, i64* %3, align 8
  %4 = alloca i8, i64 4, align 4
  %5 = alloca i8, i64 4, align 4
  %6 = alloca i8, i64 4, align 4
  %7 = alloca i8, i64 4, align 4
  %8 = alloca i8, i64 4, align 4
  %9 = call i32 @__nvvm_thread_idx_x()
  %10 = call i32 @__nvvm_block_idx_x()
  %11 = call i32 @__nvvm_block_dim_x()
  %12 = mul i32 %10, %11
  %13 = add i32 %12, %9
  %14 = icmp eq i32 %g, 0
  br i1 %14, label %panic, label %bb1
bb1:
  %15 = icmp eq i32 %g, -1
  %16 = icmp eq i32 %13, -2147483648
  %17 = and i1 %15, %16
  br i1 %17, label %panic, label %bb2
bb2:
  %18 = sdiv i32 %13, %g
  br i1 %14, label %panic, label %bb3
bb3:
  br i1 %17, label %panic, label %bb4
bb4:
  %19 = srem i32 %13, %g
  %20 = icmp slt i32 %18, %g
  br i1 %20, label %bb5, label %bb7
bb7:
  br label %bb31
bb5:
  %21 = icmp slt i32 %19, %g
  br i1 %21, label %bb6, label %bb7
bb6:
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4)
  %22 = bitcast i8* %4 to i32*
  store i32 0, i32* %22, align 4
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5)
  %23 = mul i32 %18, %g
  %24 = bitcast i8* %5 to i32*
  store i32 %23, i32* %24, align 4
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6)
  %25 = bitcast i8* %6 to i32*
  store i32 %19, i32* %25, align 4
  %26 = bitcast i8* %5 to i32*
  %27 = load i32, i32* %26, align 4
  %28 = bitcast i8* %6 to i32*
  %29 = load i32, i32* %28, align 4
  %30 = add i32 %27, %29
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7)
  %31 = bitcast i8* %7 to i32*
  store i32 0, i32* %31, align 4
  %32 = sdiv i32 %g, 8
  br label %bb8
bb31:
  ret void
bb8:
  %33 = bitcast i8* %7 to i32*
  %34 = load i32, i32* %33, align 4
  %35 = icmp slt i32 %34, %32
  br i1 %35, label %bb9, label %bb26
bb26:
  %36 = mul i32 %32, 8
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %8)
  %37 = bitcast i8* %8 to i32*
  store i32 %36, i32* %37, align 4
  br label %bb27
bb9:
  %38 = bitcast i8* %5 to i32*
  %39 = load i32, i32* %38, align 4
  %40 = sext i32 %39 to i64
  %41 = icmp ult i64 %40, %b
  br i1 %41, label %bb10, label %panic
bb27:
  %42 = bitcast i8* %8 to i32*
  %43 = load i32, i32* %42, align 4
  %44 = icmp slt i32 %43, %g
  br i1 %44, label %bb35, label %bb37
bb37:
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %8)
  %45 = sext i32 %30 to i64
  %46 = bitcast i8* %4 to i32*
  %47 = load i32, i32* %46, align 4
  %48 = bitcast i8* %0 to %adt0*
  call void @_ZN4cuda4dmem15Buffer$LT$T$GT$3set17h0322ab43b2f9fd10E(%adt0* %48, i64 %45, i32 %47)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %7)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %6)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4)
  br label %bb31
bb35:
  %49 = bitcast i8* %8 to i32*
  %50 = load i32, i32* %49, align 4
  %51 = call i32 @_ZN47_$LT$i32$u20$as$u20$core__iter__range__Step$GT$17forward_unchecked17h35defd4ccf24a22bE(i32 %50, i64 1)
  %52 = bitcast i8* %8 to i32*
  store i32 %51, i32* %52, align 4
  %53 = bitcast i8* %5 to i32*
  %54 = load i32, i32* %53, align 4
  %55 = sext i32 %54 to i64
  %56 = icmp ult i64 %55, %b
  br i1 %56, label %bb28, label %panic
bb28:
  %57 = bitcast i8* %a to [0 x i32]*
  %58 = getelementptr inbounds [0 x i32], [0 x i32]* %57,i64 0,i64 %55
  %59 = load i32, i32* %58, align 4
  %60 = bitcast i8* %6 to i32*
  %61 = load i32, i32* %60, align 4
  %62 = sext i32 %61 to i64
  %63 = icmp ult i64 %62, %d
  br i1 %63, label %bb29, label %panic
bb29:
  %64 = bitcast i8* %c to [0 x i32]*
  %65 = getelementptr inbounds [0 x i32], [0 x i32]* %64,i64 0,i64 %62
  %66 = load i32, i32* %65, align 4
  %67 = mul i32 %59, %66
  %68 = bitcast i8* %4 to i32*
  %69 = load i32, i32* %68, align 4
  %70 = add i32 %69, %67
  %71 = bitcast i8* %4 to i32*
  store i32 %70, i32* %71, align 4
  %72 = bitcast i8* %5 to i32*
  %73 = load i32, i32* %72, align 4
  %74 = add i32 %73, 1
  %75 = bitcast i8* %5 to i32*
  store i32 %74, i32* %75, align 4
  %76 = bitcast i8* %6 to i32*
  %77 = load i32, i32* %76, align 4
  %78 = add i32 %77, %g
  %79 = bitcast i8* %6 to i32*
  store i32 %78, i32* %79, align 4
  br label %bb27
bb10:
  %80 = bitcast i8* %a to [0 x i32]*
  %81 = getelementptr inbounds [0 x i32], [0 x i32]* %80,i64 0,i64 %40
  %82 = load i32, i32* %81, align 4
  %83 = bitcast i8* %6 to i32*
  %84 = load i32, i32* %83, align 4
  %85 = sext i32 %84 to i64
  %86 = icmp ult i64 %85, %d
  br i1 %86, label %bb11, label %panic
bb11:
  %87 = bitcast i8* %c to [0 x i32]*
  %88 = getelementptr inbounds [0 x i32], [0 x i32]* %87,i64 0,i64 %85
  %89 = load i32, i32* %88, align 4
  %90 = mul i32 %82, %89
  %91 = bitcast i8* %4 to i32*
  %92 = load i32, i32* %91, align 4
  %93 = add i32 %90, %92
  %94 = bitcast i8* %4 to i32*
  store i32 %93, i32* %94, align 4
  %95 = bitcast i8* %5 to i32*
  %96 = load i32, i32* %95, align 4
  %97 = add i32 %96, 1
  %98 = sext i32 %97 to i64
  %99 = icmp ult i64 %98, %b
  br i1 %99, label %bb12, label %panic
bb12:
  %100 = bitcast i8* %a to [0 x i32]*
  %101 = getelementptr inbounds [0 x i32], [0 x i32]* %100,i64 0,i64 %98
  %102 = load i32, i32* %101, align 4
  %103 = bitcast i8* %6 to i32*
  %104 = load i32, i32* %103, align 4
  %105 = add i32 %104, %g
  %106 = sext i32 %105 to i64
  %107 = icmp ult i64 %106, %d
  br i1 %107, label %bb13, label %panic
bb13:
  %108 = bitcast i8* %c to [0 x i32]*
  %109 = getelementptr inbounds [0 x i32], [0 x i32]* %108,i64 0,i64 %106
  %110 = load i32, i32* %109, align 4
  %111 = mul i32 %102, %110
  %112 = bitcast i8* %4 to i32*
  %113 = load i32, i32* %112, align 4
  %114 = add i32 %111, %113
  %115 = bitcast i8* %4 to i32*
  store i32 %114, i32* %115, align 4
  %116 = bitcast i8* %5 to i32*
  %117 = load i32, i32* %116, align 4
  %118 = add i32 %117, 2
  %119 = sext i32 %118 to i64
  %120 = icmp ult i64 %119, %b
  br i1 %120, label %bb14, label %panic
bb14:
  %121 = bitcast i8* %a to [0 x i32]*
  %122 = getelementptr inbounds [0 x i32], [0 x i32]* %121,i64 0,i64 %119
  %123 = load i32, i32* %122, align 4
  %124 = bitcast i8* %6 to i32*
  %125 = load i32, i32* %124, align 4
  %126 = mul i32 2, %g
  %127 = add i32 %125, %126
  %128 = sext i32 %127 to i64
  %129 = icmp ult i64 %128, %d
  br i1 %129, label %bb15, label %panic
bb15:
  %130 = bitcast i8* %c to [0 x i32]*
  %131 = getelementptr inbounds [0 x i32], [0 x i32]* %130,i64 0,i64 %128
  %132 = load i32, i32* %131, align 4
  %133 = mul i32 %123, %132
  %134 = bitcast i8* %4 to i32*
  %135 = load i32, i32* %134, align 4
  %136 = add i32 %133, %135
  %137 = bitcast i8* %4 to i32*
  store i32 %136, i32* %137, align 4
  %138 = bitcast i8* %5 to i32*
  %139 = load i32, i32* %138, align 4
  %140 = add i32 %139, 3
  %141 = sext i32 %140 to i64
  %142 = icmp ult i64 %141, %b
  br i1 %142, label %bb16, label %panic
bb16:
  %143 = bitcast i8* %a to [0 x i32]*
  %144 = getelementptr inbounds [0 x i32], [0 x i32]* %143,i64 0,i64 %141
  %145 = load i32, i32* %144, align 4
  %146 = bitcast i8* %6 to i32*
  %147 = load i32, i32* %146, align 4
  %148 = mul i32 3, %g
  %149 = add i32 %147, %148
  %150 = sext i32 %149 to i64
  %151 = icmp ult i64 %150, %d
  br i1 %151, label %bb17, label %panic
bb17:
  %152 = bitcast i8* %c to [0 x i32]*
  %153 = getelementptr inbounds [0 x i32], [0 x i32]* %152,i64 0,i64 %150
  %154 = load i32, i32* %153, align 4
  %155 = mul i32 %145, %154
  %156 = bitcast i8* %4 to i32*
  %157 = load i32, i32* %156, align 4
  %158 = add i32 %155, %157
  %159 = bitcast i8* %4 to i32*
  store i32 %158, i32* %159, align 4
  %160 = bitcast i8* %5 to i32*
  %161 = load i32, i32* %160, align 4
  %162 = add i32 %161, 4
  %163 = sext i32 %162 to i64
  %164 = icmp ult i64 %163, %b
  br i1 %164, label %bb18, label %panic
bb18:
  %165 = bitcast i8* %a to [0 x i32]*
  %166 = getelementptr inbounds [0 x i32], [0 x i32]* %165,i64 0,i64 %163
  %167 = load i32, i32* %166, align 4
  %168 = bitcast i8* %6 to i32*
  %169 = load i32, i32* %168, align 4
  %170 = mul i32 4, %g
  %171 = add i32 %169, %170
  %172 = sext i32 %171 to i64
  %173 = icmp ult i64 %172, %d
  br i1 %173, label %bb19, label %panic
bb19:
  %174 = bitcast i8* %c to [0 x i32]*
  %175 = getelementptr inbounds [0 x i32], [0 x i32]* %174,i64 0,i64 %172
  %176 = load i32, i32* %175, align 4
  %177 = mul i32 %167, %176
  %178 = bitcast i8* %4 to i32*
  %179 = load i32, i32* %178, align 4
  %180 = add i32 %177, %179
  %181 = bitcast i8* %4 to i32*
  store i32 %180, i32* %181, align 4
  %182 = bitcast i8* %5 to i32*
  %183 = load i32, i32* %182, align 4
  %184 = add i32 %183, 5
  %185 = sext i32 %184 to i64
  %186 = icmp ult i64 %185, %b
  br i1 %186, label %bb20, label %panic
bb20:
  %187 = bitcast i8* %a to [0 x i32]*
  %188 = getelementptr inbounds [0 x i32], [0 x i32]* %187,i64 0,i64 %185
  %189 = load i32, i32* %188, align 4
  %190 = bitcast i8* %6 to i32*
  %191 = load i32, i32* %190, align 4
  %192 = mul i32 5, %g
  %193 = add i32 %191, %192
  %194 = sext i32 %193 to i64
  %195 = icmp ult i64 %194, %d
  br i1 %195, label %bb21, label %panic
bb21:
  %196 = bitcast i8* %c to [0 x i32]*
  %197 = getelementptr inbounds [0 x i32], [0 x i32]* %196,i64 0,i64 %194
  %198 = load i32, i32* %197, align 4
  %199 = mul i32 %189, %198
  %200 = bitcast i8* %4 to i32*
  %201 = load i32, i32* %200, align 4
  %202 = add i32 %199, %201
  %203 = bitcast i8* %4 to i32*
  store i32 %202, i32* %203, align 4
  %204 = bitcast i8* %5 to i32*
  %205 = load i32, i32* %204, align 4
  %206 = add i32 %205, 6
  %207 = sext i32 %206 to i64
  %208 = icmp ult i64 %207, %b
  br i1 %208, label %bb22, label %panic
bb22:
  %209 = bitcast i8* %a to [0 x i32]*
  %210 = getelementptr inbounds [0 x i32], [0 x i32]* %209,i64 0,i64 %207
  %211 = load i32, i32* %210, align 4
  %212 = bitcast i8* %6 to i32*
  %213 = load i32, i32* %212, align 4
  %214 = mul i32 6, %g
  %215 = add i32 %213, %214
  %216 = sext i32 %215 to i64
  %217 = icmp ult i64 %216, %d
  br i1 %217, label %bb23, label %panic
bb23:
  %218 = bitcast i8* %c to [0 x i32]*
  %219 = getelementptr inbounds [0 x i32], [0 x i32]* %218,i64 0,i64 %216
  %220 = load i32, i32* %219, align 4
  %221 = mul i32 %211, %220
  %222 = bitcast i8* %4 to i32*
  %223 = load i32, i32* %222, align 4
  %224 = add i32 %221, %223
  %225 = bitcast i8* %4 to i32*
  store i32 %224, i32* %225, align 4
  %226 = bitcast i8* %5 to i32*
  %227 = load i32, i32* %226, align 4
  %228 = add i32 %227, 7
  %229 = sext i32 %228 to i64
  %230 = icmp ult i64 %229, %b
  br i1 %230, label %bb24, label %panic
bb24:
  %231 = bitcast i8* %a to [0 x i32]*
  %232 = getelementptr inbounds [0 x i32], [0 x i32]* %231,i64 0,i64 %229
  %233 = load i32, i32* %232, align 4
  %234 = bitcast i8* %6 to i32*
  %235 = load i32, i32* %234, align 4
  %236 = mul i32 7, %g
  %237 = add i32 %235, %236
  %238 = sext i32 %237 to i64
  %239 = icmp ult i64 %238, %d
  br i1 %239, label %bb25, label %panic
bb25:
  %240 = bitcast i8* %c to [0 x i32]*
  %241 = getelementptr inbounds [0 x i32], [0 x i32]* %240,i64 0,i64 %238
  %242 = load i32, i32* %241, align 4
  %243 = mul i32 %233, %242
  %244 = bitcast i8* %4 to i32*
  %245 = load i32, i32* %244, align 4
  %246 = add i32 %243, %245
  %247 = bitcast i8* %4 to i32*
  store i32 %246, i32* %247, align 4
  %248 = bitcast i8* %5 to i32*
  %249 = load i32, i32* %248, align 4
  %250 = add i32 %249, 8
  %251 = bitcast i8* %5 to i32*
  store i32 %250, i32* %251, align 4
  %252 = mul i32 8, %g
  %253 = bitcast i8* %6 to i32*
  %254 = load i32, i32* %253, align 4
  %255 = add i32 %254, %252
  %256 = bitcast i8* %6 to i32*
  store i32 %255, i32* %256, align 4
  %257 = bitcast i8* %7 to i32*
  %258 = load i32, i32* %257, align 4
  %259 = add i32 %258, 1
  %260 = bitcast i8* %7 to i32*
  store i32 %259, i32* %260, align 4
  br label %bb8
panic:
  call void @llvm.trap()
  unreachable
}


define i32 @_ZN47_$LT$i32$u20$as$u20$core__iter__range__Step$GT$17forward_unchecked17h35defd4ccf24a22bE(i32 %a, i64 %b) {
start:
  %0 = alloca i8, i64 8, align 4
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0)
  %1 = trunc i64 %b to i32
  %2 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %1)
  %3 = extractvalue {i32, i1} %2, 0
  %4 = extractvalue {i32, i1} %2, 1
  %5 = icmp slt i32 %1, 0
  %6 = xor i1 %4, %5
  %7 = alloca i8, i64 1, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %7)
  %8 = call i1 @llvm.expect.i1(i1 %6, i1 0)
  %9 = zext i1 %8 to i8
  store i8 %9, i8* %7, align 1
  %10 = bitcast i8* %7 to i1*
  %11 = load i1, i1* %10, align 1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %7)
  br i1 %11, label %bb2, label %bb3
bb3:
  %12 = getelementptr inbounds i8, i8* %0,i64 4
  %13 = bitcast i8* %12 to i32*
  store i32 %3, i32* %13, align 4
  %14 = bitcast i8* %0 to i32*
  store i32 1, i32* %14, align 4
  %15 = getelementptr inbounds i8, i8* %0,i64 4
  %16 = bitcast i8* %15 to i32*
  %17 = load i32, i32* %16, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %0)
  ret i32 %17
bb2:
  %18 = bitcast i8** @static_1 to i32*
  %19 = load i32, i32* %18, align 4
  %20 = bitcast i8** @static_1 to i8*
  %21 = getelementptr inbounds i8, i8* %20,i64 4
  %22 = bitcast i8* %21 to i32*
  %23 = load i32, i32* %22, align 4
  %24 = bitcast i8* %0 to i32*
  store i32 %19, i32* %24, align 4
  %25 = getelementptr inbounds i8, i8* %0,i64 4
  %26 = bitcast i8* %25 to i32*
  store i32 %23, i32* %26, align 4
  call void @llvm.assume(i1 0)
  call void @_ZN4core4hint21unreachable_unchecked18precondition_check17h18af1954815f4ee4E()
  unreachable
}


define i1 @_ZN4core10intrinsics8unlikely17hf2930e96276c5cd5E(i1 %a) {
start:
  ret i1 %a
}


declare void @llvm.assume(i1)
declare i1 @llvm.expect.i1(i1, i1)
declare void @llvm.lifetime.end.p0i8(i64, i8*)
declare void @llvm.trap()
declare void @llvm.lifetime.start.p0i8(i64, i8*)
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
!1 = !{void(i8*, i64, i8*, i64, i8*, i64, i32)* @_ZN12rust_kernels10matrix_mul17h4f87c4a65a00d4eeE, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0}

