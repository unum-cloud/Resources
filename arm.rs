//! NEON, streaming SVE, SME and SME2 forms as assembly text, with the scalar, atomic and memory
//! forms our projects call.
//!
//! Throughput kernels rotate over 31 vector registers, over every ZA tile of the element size, over
//! 16 ZA slices, over the 8 vector-group offsets of the ZA array, over 8 predicates or over 8 cache
//! lines. Streaming kernels enter and leave streaming mode once per call, which the two-unroll
//! difference cancels.
//!
//! See also:
//! - <https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow_aarch64.S>
//! - <https://ashvardanian.com/posts/aws-graviton-checksums-on-neon-vs-sve/>
//! - <https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/>
//! - <https://ashvardanian.com/posts/numkong/>

use std::arch::asm;

use crate::{Form, Mode, Reference};

pub const NAME: &str = "arm";

/// Names the `hw.optional.arm` sysctl that macOS reports for a Linux `/proc/cpuinfo` flag.
#[cfg(target_os = "macos")]
pub fn apple_feature(feature: &str) -> &'static str {
    match feature {
        // macOS reports no flag for the base instruction set, which every AdvSIMD core has.
        "fp" | "asimd" => "AdvSIMD",
        "asimdhp" => "FEAT_FP16",
        "asimdfhm" => "FEAT_FHM",
        "asimddp" => "FEAT_DotProd",
        "i8mm" => "FEAT_I8MM",
        "bf16" => "FEAT_BF16",
        "aes" => "FEAT_AES",
        "pmull" => "FEAT_PMULL",
        "sha2" => "FEAT_SHA256",
        "atomics" => "FEAT_LSE",
        "lrcpc" => "FEAT_LRCPC",
        "sme" => "FEAT_SME",
        "sme2" => "FEAT_SME2",
        "smef16f16" => "FEAT_SME_F16F16",
        "smef64f64" => "FEAT_SME_F64F64",
        _ => panic!("no macOS sysctl for the {feature} flag"),
    }
}

/// Register classes: the latency kernel's register, the throughput kernel's rotation, and the
/// reserved operand the rotation leaves out. Vector, predicate and general classes give register
/// numbers that templates prefix, as in `v\v\().4s`, `p\v/z` or `x\v`; ZA classes give names.
macro_rules! class {
    (v, $part:ident) => { class!(@ $part, "0",
        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, \
         25, 26, 27, 28, 29, 30", "31") };
    (z, $part:ident) => { class!(v, $part) };
    // Leaves registers 16 to 31 for table and structure lists.
    (v_low, $part:ident) => { class!(@ $part, "0",
        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15", "31") };
    (z_low, $part:ident) => { class!(v_low, $part) };
    // Compares govern with the rotated predicate, which their encoding limits to p0 to p7.
    (p, $part:ident) => { class!(@ $part, "0", "0, 1, 2, 3, 4, 5, 6, 7", "15") };
    // x0 to x7 address their own cache lines and x8 a ninth; `x1\v` is the data register beside `x\v`.
    (lines, $part:ident) => { class!(@ $part, "0", "0, 1, 2, 3, 4, 5, 6, 7", "8") };
    (x, $part:ident) => { class!(lines, $part) };
    (za_s, $part:ident) => { class!(@ $part, "za0", "za0, za1, za2, za3", "z31") };
    (za_h, $part:ident) => { class!(@ $part, "za0", "za0, za1", "z31") };
    (za_d, $part:ident) => { class!(@ $part, "za0", "za0, za1, za2, za3, za4, za5, za6, za7", "z31") };
    (za_s_horizontal, $part:ident) => { class!(@ $part, "\"za0h.s[w12, 0]\"",
        "\"za0h.s[w12, 0]\", \"za0h.s[w12, 1]\", \"za0h.s[w12, 2]\", \"za0h.s[w12, 3]\", \
         \"za1h.s[w12, 0]\", \"za1h.s[w12, 1]\", \"za1h.s[w12, 2]\", \"za1h.s[w12, 3]\", \
         \"za2h.s[w12, 0]\", \"za2h.s[w12, 1]\", \"za2h.s[w12, 2]\", \"za2h.s[w12, 3]\", \
         \"za3h.s[w12, 0]\", \"za3h.s[w12, 1]\", \"za3h.s[w12, 2]\", \"za3h.s[w12, 3]\"", "z31") };
    (za_s_vertical, $part:ident) => { class!(@ $part, "\"za0v.s[w12, 0]\"",
        "\"za0v.s[w12, 0]\", \"za0v.s[w12, 1]\", \"za0v.s[w12, 2]\", \"za0v.s[w12, 3]\", \
         \"za1v.s[w12, 0]\", \"za1v.s[w12, 1]\", \"za1v.s[w12, 2]\", \"za1v.s[w12, 3]\", \
         \"za2v.s[w12, 0]\", \"za2v.s[w12, 1]\", \"za2v.s[w12, 2]\", \"za2v.s[w12, 3]\", \
         \"za3v.s[w12, 0]\", \"za3v.s[w12, 1]\", \"za3v.s[w12, 2]\", \"za3v.s[w12, 3]\"", "z31") };
    (za_d_horizontal, $part:ident) => { class!(@ $part, "\"za0h.d[w12, 0]\"",
        "\"za0h.d[w12, 0]\", \"za0h.d[w12, 1]\", \"za1h.d[w12, 0]\", \"za1h.d[w12, 1]\", \
         \"za2h.d[w12, 0]\", \"za2h.d[w12, 1]\", \"za3h.d[w12, 0]\", \"za3h.d[w12, 1]\", \
         \"za4h.d[w12, 0]\", \"za4h.d[w12, 1]\", \"za5h.d[w12, 0]\", \"za5h.d[w12, 1]\", \
         \"za6h.d[w12, 0]\", \"za6h.d[w12, 1]\", \"za7h.d[w12, 0]\", \"za7h.d[w12, 1]\"", "z31") };
    (za_d_vertical, $part:ident) => { class!(@ $part, "\"za0v.d[w12, 0]\"",
        "\"za0v.d[w12, 0]\", \"za0v.d[w12, 1]\", \"za1v.d[w12, 0]\", \"za1v.d[w12, 1]\", \
         \"za2v.d[w12, 0]\", \"za2v.d[w12, 1]\", \"za3v.d[w12, 0]\", \"za3v.d[w12, 1]\", \
         \"za4v.d[w12, 0]\", \"za4v.d[w12, 1]\", \"za5v.d[w12, 0]\", \"za5v.d[w12, 1]\", \
         \"za6v.d[w12, 0]\", \"za6v.d[w12, 1]\", \"za7v.d[w12, 0]\", \"za7v.d[w12, 1]\"", "z31") };
    (za_h_horizontal, $part:ident) => { class!(@ $part, "\"za0h.h[w12, 0]\"",
        "\"za0h.h[w12, 0]\", \"za0h.h[w12, 1]\", \"za0h.h[w12, 2]\", \"za0h.h[w12, 3]\", \
         \"za0h.h[w12, 4]\", \"za0h.h[w12, 5]\", \"za0h.h[w12, 6]\", \"za0h.h[w12, 7]\", \
         \"za1h.h[w12, 0]\", \"za1h.h[w12, 1]\", \"za1h.h[w12, 2]\", \"za1h.h[w12, 3]\", \
         \"za1h.h[w12, 4]\", \"za1h.h[w12, 5]\", \"za1h.h[w12, 6]\", \"za1h.h[w12, 7]\"", "z31") };
    (za_h_vertical, $part:ident) => { class!(@ $part, "\"za0v.h[w12, 0]\"",
        "\"za0v.h[w12, 0]\", \"za0v.h[w12, 1]\", \"za0v.h[w12, 2]\", \"za0v.h[w12, 3]\", \
         \"za0v.h[w12, 4]\", \"za0v.h[w12, 5]\", \"za0v.h[w12, 6]\", \"za0v.h[w12, 7]\", \
         \"za1v.h[w12, 0]\", \"za1v.h[w12, 1]\", \"za1v.h[w12, 2]\", \"za1v.h[w12, 3]\", \
         \"za1v.h[w12, 4]\", \"za1v.h[w12, 5]\", \"za1v.h[w12, 6]\", \"za1v.h[w12, 7]\"", "z31") };
    // The single-vector operand of multi-vector forms encodes only z0 to z15.
    (vgx4, $part:ident) => { class!(@ $part, "0", "0, 1, 2, 3, 4, 5, 6, 7", "z15") };
    (@ first, $first:literal, $rotated:literal, $fixed:literal) => { $first };
    (@ rotated, $first:literal, $rotated:literal, $fixed:literal) => { $rotated };
    (@ fixed, $first:literal, $rotated:literal, $fixed:literal) => { $fixed };
}

/// Extensions: the CSV stem, the mode, the reference, and the code around the
/// loop. Every prologue points x0 to x8 at cache lines that hold their own addresses, enters the
/// mode, and fills every vector register with `0x3f` bytes, finite and normal in every type, and
/// every predicate with true lanes.
macro_rules! extension {
    (neon, $part:ident) => { extension!(@core $part, "neon") };
    (fp16, $part:ident) => { extension!(@core $part, "fp16") };
    (fp16fml, $part:ident) => { extension!(@core $part, "fp16fml") };
    (bf16, $part:ident) => { extension!(@core $part, "bf16") };
    (dotprod, $part:ident) => { extension!(@core $part, "dotprod") };
    (i8mm, $part:ident) => { extension!(@core $part, "i8mm") };
    (aes, $part:ident) => { extension!(@core $part, "aes") };
    (sha2, $part:ident) => { extension!(@core $part, "sha2") };
    (base, $part:ident) => { extension!(@core $part, "base") };
    (atomics, $part:ident) => { extension!(@core $part, "atomics") };
    (lse, $part:ident) => { extension!(@core $part, "lse") };
    (rcpc, $part:ident) => { extension!(@core $part, "rcpc") };
    (sve_streaming, $part:ident) => {
        extension!(@streaming $part, "sve-streaming", "smstart sm", "smstop sm")
    };
    (sve2_streaming, $part:ident) => {
        extension!(@streaming $part, "sve2-streaming", "smstart sm", "smstop sm")
    };
    (sme, $part:ident) => { extension!(@streaming $part, "sme", extension!(@za), "smstop") };
    (sme2, $part:ident) => { extension!(@streaming $part, "sme2", extension!(@za), "smstop") };
    (sme_f16f16, $part:ident) => {
        extension!(@streaming $part, "sme-f16f16", extension!(@za), "smstop")
    };
    (sme_f64f64, $part:ident) => {
        extension!(@streaming $part, "sme-f64f64", extension!(@za), "smstop")
    };
    (@core $part:ident, $stem:literal) => { extension!(@ $part, $stem, (crate::Mode::Standard), (&CORE_REFERENCE),
        concat!(extension!(@lines), extension!(@fill_v)), "") };
    (@streaming $part:ident, $stem:literal, $entry:expr, $exit:literal) => {
        extension!(@ $part, $stem, (crate::Mode::Streaming), (&STREAMING_REFERENCE),
            concat!(extension!(@lines), $entry, "\n", extension!(@fill_z)), $exit)
    };
    // w11 selects ZA vector groups and w12 ZA slices.
    (@za) => { "smstart\nmov w11, #0\nmov w12, #0" };
    (@lines) => { concat!(".set line_offset, 0\n.irp register, ", class!(lines, rotated), ", ",
        class!(lines, fixed), "\n", r"add x\register, x10, #line_offset", "\n",
        r"str x\register, [x\register]", "\n", ".set line_offset, line_offset + {line_bytes}\n.endr\n") };
    (@fill_v) => { concat!(".irp register, ", class!(v, rotated), ", ", class!(v, fixed), "\n",
        r"movi v\register\().16b, #0x3f", "\n.endr") };
    (@fill_z) => { concat!(".irp register, ", class!(z, rotated), ", ", class!(z, fixed), "\n",
        r"dup z\register\().b, #0x3f", "\n.endr\n",
        ".irp register, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\n",
        r"ptrue p\register\().b", "\n.endr") };
    (@ stem, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $stem };
    (@ mode, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $mode };
    (@ reference, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $reference };
    (@ prologue, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $prologue };
    (@ epilogue, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $epilogue };
}

/// The `UNROLL` and `2 × UNROLL` kernels of a step template, each copy taking the next entry of
/// `registers` as `\v` and `fixed` as `\fixed`.
macro_rules! kernels {
    ($extension:ident, $template:literal, $registers:expr, $fixed:expr) => {{
        fn kernel<const COPIES: usize>(iterations: u64) {
            let mut lines = [crate::Line::default(); crate::registers(class!(lines, rotated)) + 1];
            unsafe {
                asm!(
                    ".arch_extension fp16",
                    ".arch_extension fp16fml",
                    ".arch_extension dotprod",
                    ".arch_extension i8mm",
                    ".arch_extension bf16",
                    ".arch_extension aes",
                    ".arch_extension sha2",
                    ".arch_extension lse",
                    ".arch_extension rcpc",
                    ".arch_extension sme2",
                    ".arch_extension sme-f16f16",
                    ".arch_extension sme-f64f64",
                    ".macro step v, fixed",
                    $template,
                    ".endm",
                    extension!($extension, prologue),
                    "2:",
                    ".rept {rotations}",
                    concat!(".irp register, ", $registers),
                    concat!(r#"step "\register", ""#, $fixed, "\""),
                    ".endr",
                    ".endr",
                    "subs x9, x9, #1",
                    "b.ne 2b",
                    extension!($extension, epilogue),
                    ".purgem step",
                    line_bytes = const size_of::<crate::Line>(),
                    rotations = const COPIES / crate::registers($registers),
                    inout("x9") iterations => _,
                    inout("x10") lines.as_mut_ptr() => _,
                    // The C ABI preserves the low halves of these, which the vector prologues overwrite.
                    out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                    out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                    clobber_abi("C"),
                    options(nostack),
                )
            }
        }
        [kernel::<UNROLL> as crate::Kernel, kernel::<{ 2 * UNROLL }>]
    }};
}

pub static CORE_REFERENCE: Reference = Reference {
    name: "add x",
    kernels: kernels!(neon, r"add \v, \v, #1", "x11", "x11"),
};

pub static STREAMING_REFERENCE: Reference = Reference {
    name: "add z.s",
    kernels: kernels!(
        sve_streaming,
        r"add z\v\().s, z\v\().s, z\fixed\().s",
        class!(z, first),
        class!(z, fixed)
    ),
};

forms! {
//  mnemonic  form                                         input_type output_type extension      feature   class           kernels     template                                                              intrinsics
    DUP       "v.16b, w"                                   "u8"   "u8"   neon           asimd     v               throughput  r"dup v\v\().16b, w8"                                                 "vdupq_n_s8 vdupq_n_u8";
    LDR       "q, [x]"                                     "u128" "u128" neon           asimd     lines           throughput  r"ldr q\v, [x\v]"                                                     "vld1q_bf16 vld1q_f32 vld1q_f64 vld1q_s16 vld1q_s32 vld1q_s64 vld1q_s8 vld1q_u16 vld1q_u32 vld1q_u64 vld1q_u8";
    AND       "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"and v\v\().16b, v\v\().16b, v\fixed\().16b"                         "vandq_s32 vandq_s64 vandq_u16 vandq_u32 vandq_u64 vandq_u8";
    BSL       "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"bsl v\v\().16b, v\fixed\().16b, v\fixed\().16b"                     "vbslq_f16 vbslq_f32 vbslq_f64 vbslq_s16 vbslq_s32 vbslq_s64 vbslq_s8 vbslq_u16 vbslq_u32 vbslq_u64 vbslq_u8";
    DUP       "v.4s, v.s[imm]"                             "u32"  "u32"  neon           asimd     v               chained     r"dup v\v\().4s, v\v\().s[1]"                                         "vdupq_laneq_f32 vdupq_laneq_u32 vdupq_n_f32";
    CMEQ      "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"cmeq v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vceqq_s8 vceqq_u8";
    DUP       "v.4s, w"                                    "u32"  "u32"  neon           asimd     v               throughput  r"dup v\v\().4s, w8"                                                  "vdupq_n_s32 vdupq_n_u32";
    ORR       "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"orr v\v\().16b, v\v\().16b, v\fixed\().16b"                         "vorrq_s32 vorrq_u16 vorrq_u32 vorrq_u64 vorrq_u8";
    FMLA      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmla v\v\().4s, v\fixed\().4s, v\fixed\().4s"                       "vfmaq_f32 vmlaq_f32";
    DUP       "v.2d, v.d[imm]"                             "u64"  "u64"  neon           asimd     v               chained     r"dup v\v\().2d, v\v\().d[1]"                                         "vdupq_n_f64";
    STR       "q, [x]"                                     "u128" "u128" neon           asimd     lines           throughput  r"str q\v, [x\v]"                                                     "vst1q_f32 vst1q_f64 vst1q_s16 vst1q_s32 vst1q_s64 vst1q_s8 vst1q_u16 vst1q_u32 vst1q_u64 vst1q_u8";
    FMLA      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fmla v\v\().2d, v\fixed\().2d, v\fixed\().2d"                       "vfmaq_f64";
    FADDP     "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"faddp v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vaddvq_f32 vpaddq_f32";
    DUP       "v.8h, w"                                    "u16"  "u16"  neon           asimd     v               throughput  r"dup v\v\().8h, w8"                                                  "vdupq_n_s16 vdupq_n_u16";
    FCVTL     "v.4s, v.4h"                                 "f16"  "f32"  neon           asimd     v               chained     r"fcvtl v\v\().4s, v\v\().4h"                                         "vcvt_f32_f16";
    LDR       "d, [x]"                                     "u64"  "u64"  neon           asimd     lines           throughput  r"ldr d\v, [x\v]"                                                     "vld1_bf16 vld1_f32 vld1_s16 vld1_s32 vld1_s8 vld1_u16 vld1_u32 vld1_u8";
    FMUL      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmul v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vmulq_f32";
    FMOV      "d, x"                                       "u64"  "u64"  neon           asimd     v               throughput  r"fmov d\v, x8"                                                       "vcreate_p64 vcreate_s8 vcreate_u16 vcreate_u32 vcreate_u64 vcreate_u8";
    ADD       "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"add v\v\().4s, v\v\().4s, v\fixed\().4s"                            "vaddq_s32 vaddq_u32 vshlq_n_s32 vshlq_n_u32";
    FADDP     "s, v.2s"                                    "f32"  "f32"  neon           asimd     v               chained     r"faddp s\v, v\v\().2s"                                               "vaddvq_f32";
    FCVTL2    "v.4s, v.8h"                                 "f16"  "f32"  neon           asimd     v               chained     r"fcvtl2 v\v\().4s, v\v\().8h"                                        "vcvt_high_f32_f16";
    FADD      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fadd v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vaddq_f64";
    DUP       "v.2d, x"                                    "u64"  "u64"  neon           asimd     v               throughput  r"dup v\v\().2d, x8"                                                  "vdupq_n_s64 vdupq_n_u64";
    ADD       "v.2d, v.2d, v.2d"                           "i64"  "i64"  neon           asimd     v               chained     r"add v\v\().2d, v\v\().2d, v\fixed\().2d"                            "vaddq_s64 vaddq_u64 vshlq_n_u64";
    CMHI      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"cmhi v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vcgtq_u8 vcltq_u8";
    FSUB      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fsub v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vsubq_f64";
    ADD       "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"add v\v\().16b, v\v\().16b, v\fixed\().16b"                         "vaddq_u8 vshlq_n_s8 vshlq_n_u8";
    FADD      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fadd v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vaddq_f32";
    MOV       "v.d[imm], v.d[imm]"                         "u64"  "u64"  neon           asimd     v               chained     r"mov v\v\().d[1], v\fixed\().d[1]"                                   "vcombine_f16 vcombine_f32 vcombine_s16 vcombine_s8 vcombine_u16 vcombine_u32 vcombine_u64 vcombine_u8 vsetq_lane_f64";
    EOR       "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"eor v\v\().16b, v\v\().16b, v\fixed\().16b"                         "veorq_s16 veorq_s64 veorq_u16 veorq_u32 veorq_u64 veorq_u8";
    MOV       "x, v.d[imm]"                                "u64"  "u64"  neon           asimd     v               throughput  r"mov x8, v\v\().d[1]"                                                "vgetq_lane_s64 vgetq_lane_u64";
    FMUL      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fmul v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vmulq_f64";
    USHLL     "v.4s, v.4h, #imm"                           "u16"  "u32"  neon           asimd     v               chained     r"ushll v\v\().4s, v\v\().4h, #0"                                     "vmovl_u16 vshll_n_u16";
    CMHS      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"cmhs v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vcgeq_u8 vcleq_u8";
    UADDLP    "v.2d, v.4s"                                 "u32"  "u64"  neon           asimd     v               chained     r"uaddlp v\v\().2d, v\v\().4s"                                        "vpaddlq_u32";
    EXT       "v.16b, v.16b, v.16b, #imm"                  "u8"   "u8"   neon           asimd     v               chained     r"ext v\v\().16b, v\v\().16b, v\fixed\().16b, #1"                     "vextq_u16 vextq_u32 vextq_u64 vextq_u8";
    FCVTL     "v.2d, v.2s"                                 "f32"  "f64"  neon           asimd     v               chained     r"fcvtl v\v\().2d, v\v\().2s"                                         "vcvt_f64_f32";
    UMINV     "b, v.16b"                                   "u8"   "u8"   neon           asimd     v               chained     r"uminv b\v, v\v\().16b"                                              "vminvq_u8";
    STR       "d, [x]"                                     "u64"  "u64"  neon           asimd     lines           throughput  r"str d\v, [x\v]"                                                     "vst1_bf16 vst1_f32 vst1_s16 vst1_s32 vst1_s8 vst1_u16 vst1_u32 vst1_u8";
    SUB       "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"sub v\v\().16b, v\v\().16b, v\fixed\().16b"                         "vsubq_u8";
    FMUL      "v.4s, v.4s, v.s[imm]"                       "f32"  "f32"  neon           asimd     v               chained     r"fmul v\v\().4s, v\v\().4s, v\fixed\().s[1]"                         "vmulq_laneq_f32 vmulq_n_f32";
    USHR      "v.16b, v.16b, #imm"                         "u8"   "u8"   neon           asimd     v               chained     r"ushr v\v\().16b, v\v\().16b, #1"                                    "vshrq_n_u8";
    USHLL     "v.8h, v.8b, #imm"                           "u8"   "u16"  neon           asimd     v               chained     r"ushll v\v\().8h, v\v\().8b, #0"                                     "vmovl_u8 vshll_n_u8";
    BIC       "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"bic v\v\().16b, v\v\().16b, v\fixed\().16b"                         "vbicq_u16 vbicq_u32 vbicq_u8";
    FSUB      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fsub v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vsubq_f32";
    MOV       "d, v.d[imm]"                                "u64"  "u64"  neon           asimd     v               chained     r"mov d\v, v\v\().d[1]"                                               "vget_high_p64 vget_high_u16 vget_high_u32 vget_high_u8 vgetq_lane_f64";
    TBL       "v.16b, {v.16b}, v.16b"                      "u8"   "u8"   neon           asimd     v               chained     r"tbl v\v\().16b, {{v\v\().16b}}, v\fixed\().16b"                     "vqtbl1q_s8 vqtbl1q_u8";
    CMEQ      "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"cmeq v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vceqq_s16 vceqq_u16";
    LD1       "{v.16b, v.16b, v.16b, v.16b}, [x]"          "u8"   "u8"   neon           asimd     lines           throughput  r"ld1 {{v16.16b, v17.16b, v18.16b, v19.16b}}, [x\v]"                  "vld1q_s8_x4 vld1q_u8_x4";
    ADDV      "s, v.4s"                                    "i32"  "i32"  neon           asimd     v               chained     r"addv s\v, v\v\().4s"                                                "vaddvq_s32 vaddvq_u32";
    CMEQ      "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"cmeq v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vceqq_s32 vceqq_u32";
    UMAXV     "b, v.16b"                                   "u8"   "u8"   neon           asimd     v               chained     r"umaxv b\v, v\v\().16b"                                              "vmaxvq_u8";
    FMAX      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmax v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vmaxq_f32";
    TBL       "v.16b, {v.16b, v.16b, v.16b, v.16b}, v.16b" "u8"   "u8"   neon           asimd     v_low           chained     r"tbl v\v\().16b, {{v16.16b, v17.16b, v18.16b, v19.16b}}, v\v\().16b" "vqtbl4q_s8 vqtbl4q_u8";
    UADDLP    "v.4s, v.8h"                                 "u16"  "u32"  neon           asimd     v               chained     r"uaddlp v\v\().4s, v\v\().8h"                                        "vpaddlq_u16";
    ADD       "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"add v\v\().8h, v\v\().8h, v\fixed\().8h"                            "vaddq_s16 vaddq_u16 vshlq_n_u16";
    CMTST     "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"cmtst v\v\().16b, v\v\().16b, v\fixed\().16b"                       "vtstq_u8";
    FADDP     "d, v.2d"                                    "f64"  "f64"  neon           asimd     v               chained     r"faddp d\v, v\v\().2d"                                               "vaddvq_f64";
    DUP       "v.2s, v.s[imm]"                             "u32"  "u32"  neon           asimd     v               chained     r"dup v\v\().2s, v\v\().s[1]"                                         "vdup_n_f32";
    ADDP      "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"addp v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vpaddq_s32 vpaddq_u32";
    UCVTF     "v.4s, v.4s"                                 "u32"  "f32"  neon           asimd     v               chained     r"ucvtf v\v\().4s, v\v\().4s"                                         "vcvtq_f32_u32";
    SCVTF     "v.4s, v.4s"                                 "i32"  "f32"  neon           asimd     v               chained     r"scvtf v\v\().4s, v\v\().4s"                                         "vcvtq_f32_s32";
    TBL       "v.16b, {v.16b, v.16b}, v.16b"               "u8"   "u8"   neon           asimd     v_low           chained     r"tbl v\v\().16b, {{v16.16b, v17.16b}}, v\v\().16b"                   "vqtbl2q_u8";
    CMGT      "v.2d, v.2d, v.2d"                           "i64"  "i64"  neon           asimd     v               chained     r"cmgt v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vcgtq_s64 vcltq_s64";
    FCVTL2    "v.2d, v.4s"                                 "f32"  "f64"  neon           asimd     v               chained     r"fcvtl2 v\v\().2d, v\v\().4s"                                        "vcvt_high_f64_f32";
    MOV       "s, v.s[imm]"                                "u32"  "u32"  neon           asimd     v               chained     r"mov s\v, v\v\().s[1]"                                               "vget_lane_f32 vgetq_lane_f32";
    USHR      "v.4s, v.4s, #imm"                           "u32"  "u32"  neon           asimd     v               chained     r"ushr v\v\().4s, v\v\().4s, #1"                                      "vshrq_n_u32";
    UMINV     "s, v.4s"                                    "u32"  "u32"  neon           asimd     v               chained     r"uminv s\v, v\v\().4s"                                               "vminvq_u32";
    USHLL2    "v.4s, v.8h, #imm"                           "u16"  "u32"  neon           asimd     v               chained     r"ushll2 v\v\().4s, v\v\().8h, #0"                                    "vmovl_high_u16 vshll_high_n_u16";
    XTN       "v.4h, v.4s"                                 "i32"  "i16"  neon           asimd     v               chained     r"xtn v\v\().4h, v\v\().4s"                                           "vmovn_s32 vmovn_u32";
    SUB       "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"sub v\v\().4s, v\v\().4s, v\fixed\().4s"                            "vsubq_s32 vsubq_u32";
    FMLS      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fmls v\v\().2d, v\fixed\().2d, v\fixed\().2d"                       "vfmsq_f64";
    FMLS      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmls v\v\().4s, v\fixed\().4s, v\fixed\().4s"                       "vfmsq_f32";
    SMAX      "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"smax v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vmaxq_s32";
    SUB       "v.2d, v.2d, v.2d"                           "i64"  "i64"  neon           asimd     v               chained     r"sub v\v\().2d, v\v\().2d, v\fixed\().2d"                            "vsubq_s64 vsubq_u64";
    CMHI      "v.2d, v.2d, v.2d"                           "u64"  "u64"  neon           asimd     v               chained     r"cmhi v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vcgtq_u64 vcltq_u64";
    CMHI      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"cmhi v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vcgtq_u32 vcltq_u32";
    FCVTN     "v.4h, v.4s"                                 "f32"  "f16"  neon           asimd     v               chained     r"fcvtn v\v\().4h, v\v\().4s"                                         "vcvt_f16_f32";
    FSQRT     "v.2s, v.2s"                                 "f32"  "f32"  neon           asimd     v               chained     r"fsqrt v\v\().2s, v\v\().2s"                                         "vsqrt_f32";
    LD1R      "{v.4h}, [x]"                                "u16"  "u16"  neon           asimd     lines           throughput  r"ld1r {{v\v\().4h}}, [x\v]"                                          "vld1_dup_f16 vld1_dup_u16";
    UMINV     "h, v.8h"                                    "u16"  "u16"  neon           asimd     v               chained     r"uminv h\v, v\v\().8h"                                               "vminvq_u16";
    XTN       "v.8b, v.8h"                                 "i16"  "i8"   neon           asimd     v               chained     r"xtn v\v\().8b, v\v\().8h"                                           "vmovn_s16 vmovn_u16";
    CMGT      "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"cmgt v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vcgtq_s8 vcltq_s8";
    FNEG      "v.2d, v.2d"                                 "f64"  "f64"  neon           asimd     v               chained     r"fneg v\v\().2d, v\v\().2d"                                          "vnegq_f64";
    MOV       "w, v.s[imm]"                                "u32"  "u32"  neon           asimd     v               throughput  r"mov w8, v\v\().s[1]"                                                "vget_lane_u32";
    SADDLP    "v.4s, v.8h"                                 "i16"  "i32"  neon           asimd     v               chained     r"saddlp v\v\().4s, v\v\().8h"                                        "vpaddlq_s16";
    FMAX      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fmax v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vmaxq_f64";
    LD2       "{v.16b, v.16b}, [x]"                        "u8"   "u8"   neon           asimd     lines           throughput  r"ld2 {{v16.16b, v17.16b}}, [x\v]"                                    "vld2q_s8 vld2q_u8";
    SSHR      "v.16b, v.16b, #imm"                         "i8"   "i8"   neon           asimd     v               chained     r"sshr v\v\().16b, v\v\().16b, #1"                                    "vshrq_n_s8";
    UCVTF     "v.2d, v.2d"                                 "u64"  "f64"  neon           asimd     v               chained     r"ucvtf v\v\().2d, v\v\().2d"                                         "vcvtq_f64_u64";
    CMGT      "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"cmgt v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vcgtq_s32 vcltq_s32";
    CMHI      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"cmhi v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vcgtq_u16 vcltq_u16";
    FCVTNS    "v.4s, v.4s"                                 "f32"  "i32"  neon           asimd     v               chained     r"fcvtns v\v\().4s, v\v\().4s"                                        "vcvtnq_s32_f32";
    FABS      "v.4s, v.4s"                                 "f32"  "f32"  neon           asimd     v               chained     r"fabs v\v\().4s, v\v\().4s"                                          "vabsq_f32";
    FCMGT     "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fcmgt v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vcgtq_f32 vcltq_f32";
    LD1       "{v.16b, v.16b}, [x]"                        "u8"   "u8"   neon           asimd     lines           throughput  r"ld1 {{v16.16b, v17.16b}}, [x\v]"                                    "vld1q_u8_x2";
    LD4       "{v.16b, v.16b, v.16b, v.16b}, [x]"          "u8"   "u8"   neon           asimd     lines           throughput  r"ld4 {{v16.16b, v17.16b, v18.16b, v19.16b}}, [x\v]"                  "vld4q_s8 vld4q_u8";
    MVN       "v.16b, v.16b"                               "u8"   "u8"   neon           asimd     v               chained     r"mvn v\v\().16b, v\v\().16b"                                         "vmvnq_s16 vmvnq_s32 vmvnq_u32 vmvnq_u8";
    SMULL     "v.8h, v.8b, v.8b"                           "i8"   "i16"  neon           asimd     v               chained     r"smull v\v\().8h, v\v\().8b, v\fixed\().8b"                          "vmull_s8";
    USHR      "v.8h, v.8h, #imm"                           "u16"  "u16"  neon           asimd     v               chained     r"ushr v\v\().8h, v\v\().8h, #1"                                      "vshrq_n_u16";
    CMEQ      "v.2d, v.2d, v.2d"                           "i64"  "i64"  neon           asimd     v               chained     r"cmeq v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vceqq_s64 vceqq_u64";
    FDIV      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fdiv v\v\().4s, v\v\().4s, v\v\().4s"                               "vdivq_f32";
    FMLA      "v.4s, v.4s, v.s[imm]"                       "f32"  "f32"  neon           asimd     v               chained     r"fmla v\v\().4s, v\fixed\().4s, v\fixed\().s[1]"                     "vfmaq_laneq_f32 vfmaq_n_f32";
    FMUL      "v.2d, v.2d, v.d[imm]"                       "f64"  "f64"  neon           asimd     v               chained     r"fmul v\v\().2d, v\v\().2d, v\fixed\().d[1]"                         "vmulq_n_f64";
    LD3       "{v.16b, v.16b, v.16b}, [x]"                 "u8"   "u8"   neon           asimd     lines           throughput  r"ld3 {{v16.16b, v17.16b, v18.16b}}, [x\v]"                           "vld3q_s8 vld3q_u8";
    SMULL2    "v.8h, v.16b, v.16b"                         "i8"   "i16"  neon           asimd     v               chained     r"smull2 v\v\().8h, v\v\().16b, v\fixed\().16b"                       "vmull_high_s8";
    XTN       "v.2s, v.2d"                                 "i64"  "i32"  neon           asimd     v               chained     r"xtn v\v\().2s, v\v\().2d"                                           "vmovn_u64";
    FSQRT     "v.4s, v.4s"                                 "f32"  "f32"  neon           asimd     v               chained     r"fsqrt v\v\().4s, v\v\().4s"                                         "vsqrtq_f32";
    LD1R      "{v.2s}, [x]"                                "u32"  "u32"  neon           asimd     lines           throughput  r"ld1r {{v\v\().2s}}, [x\v]"                                          "vld1_dup_f32 vld1_dup_u32";
    LD2       "{v.4h, v.4h}, [x]"                          "u16"  "u16"  neon           asimd     lines           throughput  r"ld2 {{v16.4h, v17.4h}}, [x\v]"                                      "vld2_s16";
    SADALP    "v.4s, v.8h"                                 "i16"  "i32"  neon           asimd     v               chained     r"sadalp v\v\().4s, v\fixed\().8h"                                    "vpadalq_s16";
    SMIN      "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"smin v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vminq_s32";
    SSHLL     "v.8h, v.8b, #imm"                           "i8"   "i16"  neon           asimd     v               chained     r"sshll v\v\().8h, v\v\().8b, #0"                                     "vmovl_s8";
    FCMEQ     "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fcmeq v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vceqq_f32";
    FDIV      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fdiv v\v\().2d, v\v\().2d, v\v\().2d"                               "vdivq_f64";
    LD1R      "{v.16b}, [x]"                               "u8"   "u8"   neon           asimd     lines           throughput  r"ld1r {{v\v\().16b}}, [x\v]"                                         "vld1q_dup_u8";
    SCVTF     "v.2d, v.2d"                                 "i64"  "f64"  neon           asimd     v               chained     r"scvtf v\v\().2d, v\v\().2d"                                         "vcvtq_f64_s64";
    ADDP      "d, v.2d"                                    "i64"  "i64"  neon           asimd     v               chained     r"addp d\v, v\v\().2d"                                                "vaddvq_s64 vaddvq_u64";
    FCMGT     "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fcmgt v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vcgtq_f64 vcltq_f64";
    FMIN      "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fmin v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vminq_f64";
    LD2       "{v.4s, v.4s}, [x]"                          "u32"  "u32"  neon           asimd     lines           throughput  r"ld2 {{v16.4s, v17.4s}}, [x\v]"                                      "vld2q_f32 vld2q_s32 vld2q_u32";
    SMULL     "v.4s, v.4h, v.4h"                           "i16"  "i32"  neon           asimd     v               chained     r"smull v\v\().4s, v\v\().4h, v\fixed\().4h"                          "vmull_s16";
    SMULL2    "v.4s, v.8h, v.8h"                           "i16"  "i32"  neon           asimd     v               chained     r"smull2 v\v\().4s, v\v\().8h, v\fixed\().8h"                         "vmull_high_s16";
    ST1       "{v.s}[imm], [x]"                            "u32"  "u32"  neon           asimd     lines           throughput  r"st1 {{v\v\().s}}[1], [x\v]"                                         "vst1_lane_u32";
    UADDLV    "h, v.16b"                                   "u8"   "u16"  neon           asimd     v               chained     r"uaddlv h\v, v\v\().16b"                                             "vaddlvq_u8";
    UQADD     "v.2d, v.2d, v.2d"                           "u64"  "u64"  neon           asimd     v               chained     r"uqadd v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vqaddq_u64";
    ZIP1      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"zip1 v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vzip1q_u8 vzipq_u8";
    ZIP2      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"zip2 v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vzip2q_u8 vzipq_u8";
    CMGT      "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"cmgt v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vcgtq_s16 vcltq_s16";
    CNT       "v.16b, v.16b"                               "u8"   "u8"   neon           asimd     v               chained     r"cnt v\v\().16b, v\v\().16b"                                         "vcntq_u8";
    FMIN      "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmin v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vminq_f32";
    FMLA      "v.2d, v.2d, v.d[imm]"                       "f64"  "f64"  neon           asimd     v               chained     r"fmla v\v\().2d, v\fixed\().2d, v\fixed\().d[1]"                     "vfmaq_n_f64";
    FSQRT     "d, d"                                       "f64"  "f64"  neon           asimd     v               chained     r"fsqrt d\v, d\v"                                                     "vsqrt_f64";
    LD1R      "{v.2d}, [x]"                                "u64"  "u64"  neon           asimd     lines           throughput  r"ld1r {{v\v\().2d}}, [x\v]"                                          "vld1q_dup_f64";
    LD2       "{v.2d, v.2d}, [x]"                          "u64"  "u64"  neon           asimd     lines           throughput  r"ld2 {{v16.2d, v17.2d}}, [x\v]"                                      "vld2q_f64 vld2q_u64";
    NEG       "v.8h, v.8h"                                 "i16"  "i16"  neon           asimd     v               chained     r"neg v\v\().8h, v\v\().8h"                                           "vnegq_s16";
    SSHLL     "v.4s, v.4h, #imm"                           "i16"  "i32"  neon           asimd     v               chained     r"sshll v\v\().4s, v\v\().4h, #0"                                     "vmovl_s16";
    USHLL     "v.2d, v.2s, #imm"                           "u32"  "u64"  neon           asimd     v               chained     r"ushll v\v\().2d, v\v\().2s, #0"                                     "vmovl_u32";
    FCVTN     "v.2s, v.2d"                                 "f64"  "f32"  neon           asimd     v               chained     r"fcvtn v\v\().2s, v\v\().2d"                                         "vcvt_f32_f64";
    FMOV      "x, d"                                       "u64"  "u64"  neon           asimd     v               throughput  r"fmov x8, d\v"                                                       "vget_lane_p64 vget_lane_u64";
    ADDV      "b, v.8b"                                    "i8"   "i8"   neon           asimd     v               chained     r"addv b\v, v\v\().8b"                                                "vaddv_u8";
    FCMEQ     "v.2d, v.2d, #imm"                           "f64"  "f64"  neon           asimd     v               chained     r"fcmeq v\v\().2d, v\v\().2d, #0.0"                                   "vceqzq_f64";
    FCMEQ     "v.4s, v.4s, #imm"                           "f32"  "f32"  neon           asimd     v               chained     r"fcmeq v\v\().4s, v\v\().4s, #0.0"                                   "vceqzq_f32";
    FCVTNU    "v.4s, v.4s"                                 "f32"  "u32"  neon           asimd     v               chained     r"fcvtnu v\v\().4s, v\v\().4s"                                        "vcvtnq_u32_f32";
    NEG       "v.16b, v.16b"                               "i8"   "i8"   neon           asimd     v               chained     r"neg v\v\().16b, v\v\().16b"                                         "vnegq_s8";
    SSHR      "v.2d, v.2d, #imm"                           "i64"  "i64"  neon           asimd     v               chained     r"sshr v\v\().2d, v\v\().2d, #1"                                      "vshrq_n_s64";
    ADDP      "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"addp v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vpaddq_u8";
    DUP       "v.4h, w"                                    "u16"  "u16"  neon           asimd     v               throughput  r"dup v\v\().4h, w8"                                                  "vdup_n_s16 vdup_n_u16";
    FCVTN2    "v.8h, v.4s"                                 "f32"  "f16"  neon           asimd     v               chained     r"fcvtn2 v\v\().8h, v\fixed\().4s"                                    "vcvt_high_f16_f32";
    FSQRT     "v.2d, v.2d"                                 "f64"  "f64"  neon           asimd     v               chained     r"fsqrt v\v\().2d, v\v\().2d"                                         "vsqrtq_f64";
    LD2       "{v.8h, v.8h}, [x]"                          "u16"  "u16"  neon           asimd     lines           throughput  r"ld2 {{v16.8h, v17.8h}}, [x\v]"                                      "vld2q_s16 vld2q_u16";
    REV32     "v.16b, v.16b"                               "u8"   "u8"   neon           asimd     v               chained     r"rev32 v\v\().16b, v\v\().16b"                                       "vrev32q_u8";
    SADDLP    "v.2d, v.4s"                                 "i32"  "i64"  neon           asimd     v               chained     r"saddlp v\v\().2d, v\v\().4s"                                        "vpaddlq_s32";
    SADDLP    "v.8h, v.16b"                                "i8"   "i16"  neon           asimd     v               chained     r"saddlp v\v\().8h, v\v\().16b"                                       "vpaddlq_s8";
    SHRN      "v.8b, v.8h, #imm"                           "i16"  "i8"   neon           asimd     v               chained     r"shrn v\v\().8b, v\v\().8h, #1"                                      "vshrn_n_u16";
    SSHLL     "v.2d, v.2s, #imm"                           "i32"  "i64"  neon           asimd     v               chained     r"sshll v\v\().2d, v\v\().2s, #0"                                     "vmovl_s32";
    SSHR      "v.4s, v.4s, #imm"                           "i32"  "i32"  neon           asimd     v               chained     r"sshr v\v\().4s, v\v\().4s, #1"                                      "vshrq_n_s32";
    UADDLP    "v.8h, v.16b"                                "u8"   "u16"  neon           asimd     v               chained     r"uaddlp v\v\().8h, v\v\().16b"                                       "vpaddlq_u8";
    MUL       "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"mul v\v\().4s, v\v\().4s, v\fixed\().4s"                            "vmulq_s32 vmulq_n_s32";
    UMULL     "v.2d, v.2s, v.2s"                           "u32"  "u64"  neon           asimd     v               chained     r"umull v\v\().2d, v\v\().2s, v\fixed\().2s"                          "vmull_u32 vmull_n_u32";
    DUP       "v.8b, w"                                    "u8"   "u8"   neon           asimd     v               throughput  r"dup v\v\().8b, w8"                                                  "vdup_n_u8";
    FCVTN2    "v.4s, v.2d"                                 "f64"  "f32"  neon           asimd     v               chained     r"fcvtn2 v\v\().4s, v\fixed\().2d"                                    "vcvt_high_f32_f64";
    FCVTNS    "v.2d, v.2d"                                 "f64"  "i64"  neon           asimd     v               chained     r"fcvtns v\v\().2d, v\v\().2d"                                        "vcvtnq_s64_f64";
    FMAXNM    "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmaxnm v\v\().4s, v\v\().4s, v\fixed\().4s"                         "vmaxnmq_f32";
    LD3       "{v.4s, v.4s, v.4s}, [x]"                    "u32"  "u32"  neon           asimd     lines           throughput  r"ld3 {{v16.4s, v17.4s, v18.4s}}, [x\v]"                              "vld3q_f32 vld3q_s32 vld3q_u32";
    USHLL2    "v.8h, v.16b, #imm"                          "u8"   "u16"  neon           asimd     v               chained     r"ushll2 v\v\().8h, v\v\().16b, #0"                                   "vshll_high_n_u8";
    SMLAL     "v.4s, v.4h, v.4h"                           "i16"  "i32"  neon           asimd     v               chained     r"smlal v\v\().4s, v\fixed\().4h, v\fixed\().4h"                      "vmlal_s16 vmlal_n_s16";
    SMLAL2    "v.4s, v.8h, v.8h"                           "i16"  "i32"  neon           asimd     v               chained     r"smlal2 v\v\().4s, v\fixed\().8h, v\fixed\().8h"                     "vmlal_high_s16 vmlal_high_n_s16";
    FCMEQ     "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fcmeq v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vceqq_f64";
    FMAXV     "s, v.4s"                                    "f32"  "f32"  neon           asimd     v               chained     r"fmaxv s\v, v\v\().4s"                                               "vmaxvq_f32";
    FMINP     "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fminp v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vpminq_f32";
    FRSQRTS   "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"frsqrts v\v\().2d, v\v\().2d, v\fixed\().2d"                        "vrsqrtsq_f64";
    LD2       "{v.2s, v.2s}, [x]"                          "u32"  "u32"  neon           asimd     lines           throughput  r"ld2 {{v16.2s, v17.2s}}, [x\v]"                                      "vld2_f32";
    LD3       "{v.8h, v.8h, v.8h}, [x]"                    "u16"  "u16"  neon           asimd     lines           throughput  r"ld3 {{v16.8h, v17.8h, v18.8h}}, [x\v]"                              "vld3q_s16 vld3q_u16";
    LD4       "{v.4s, v.4s, v.4s, v.4s}, [x]"              "u32"  "u32"  neon           asimd     lines           throughput  r"ld4 {{v16.4s, v17.4s, v18.4s, v19.4s}}, [x\v]"                      "vld4q_f32 vld4q_s32 vld4q_u32";
    LD4       "{v.8h, v.8h, v.8h, v.8h}, [x]"              "u16"  "u16"  neon           asimd     lines           throughput  r"ld4 {{v16.8h, v17.8h, v18.8h, v19.8h}}, [x\v]"                      "vld4q_s16 vld4q_u16";
    SADDLV    "d, v.4s"                                    "i32"  "i64"  neon           asimd     v               chained     r"saddlv d\v, v\v\().4s"                                              "vaddlvq_s32";
    SQXTN     "v.4h, v.4s"                                 "i32"  "i16"  neon           asimd     v               chained     r"sqxtn v\v\().4h, v\v\().4s"                                         "vqmovn_s32";
    ST1       "{v.h}[imm], [x]"                            "u16"  "u16"  neon           asimd     lines           throughput  r"st1 {{v\v\().h}}[1], [x\v]"                                         "vst1_lane_f16 vst1_lane_u16";
    UMAXV     "s, v.4s"                                    "u32"  "u32"  neon           asimd     v               chained     r"umaxv s\v, v\v\().4s"                                               "vmaxvq_u32";
    UQXTN     "v.4h, v.4s"                                 "u32"  "u16"  neon           asimd     v               chained     r"uqxtn v\v\().4h, v\v\().4s"                                         "vqmovn_u32";
    UMULL2    "v.2d, v.4s, v.4s"                           "u32"  "u64"  neon           asimd     v               chained     r"umull2 v\v\().2d, v\v\().4s, v\fixed\().4s"                         "vmull_high_u32 vmull_high_n_u32";
    CMHS      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"cmhs v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vcgeq_u32 vcleq_u32";
    CMHS      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"cmhs v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vcleq_u16";
    DUP       "v.2s, w"                                    "u32"  "u32"  neon           asimd     v               throughput  r"dup v\v\().2s, w8"                                                  "vdup_n_u32";
    FMUL      "v.2s, v.2s, v.2s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmul v\v\().2s, v\v\().2s, v\fixed\().2s"                           "vmul_f32";
    SMULL     "v.2d, v.2s, v.2s"                           "i32"  "i64"  neon           asimd     v               chained     r"smull v\v\().2d, v\v\().2s, v\fixed\().2s"                          "vmull_s32";
    SMULL2    "v.2d, v.4s, v.4s"                           "i32"  "i64"  neon           asimd     v               chained     r"smull2 v\v\().2d, v\v\().4s, v\fixed\().4s"                         "vmull_high_s32";
    SRI       "v.16b, v.16b, #imm"                         "u8"   "u8"   neon           asimd     v               chained     r"sri v\v\().16b, v\fixed\().16b, #1"                                 "vsriq_n_u8";
    SSHLL2    "v.4s, v.8h, #imm"                           "i16"  "i32"  neon           asimd     v               chained     r"sshll2 v\v\().4s, v\v\().8h, #0"                                    "vmovl_high_s16";
    ST1       "{v.16b, v.16b, v.16b, v.16b}, [x]"          "u8"   "u8"   neon           asimd     lines           throughput  r"st1 {{v16.16b, v17.16b, v18.16b, v19.16b}}, [x\v]"                  "vst1q_u8_x4";
    UMIN      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"umin v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vminq_u32";
    UMULL     "v.8h, v.8b, v.8b"                           "u8"   "u16"  neon           asimd     v               chained     r"umull v\v\().8h, v\v\().8b, v\fixed\().8b"                          "vmull_u8";
    UQADD     "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"uqadd v\v\().8h, v\v\().8h, v\fixed\().8h"                          "vqaddq_u16";
    USHR      "v.2d, v.2d, #imm"                           "u64"  "u64"  neon           asimd     v               chained     r"ushr v\v\().2d, v\v\().2d, #1"                                      "vshrq_n_u64";
    ZIP1      "v.2d, v.2d, v.2d"                           "u64"  "u64"  neon           asimd     v               chained     r"zip1 v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vuzp1q_u64 vzip1q_f64";
    ZIP2      "v.2d, v.2d, v.2d"                           "u64"  "u64"  neon           asimd     v               chained     r"zip2 v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vuzp2q_u64 vzip2q_f64";
    CLZ       "v.2s, v.2s"                                 "u32"  "u32"  neon           asimd     v               chained     r"clz v\v\().2s, v\v\().2s"                                           "vclz_u32";
    FABS      "v.2d, v.2d"                                 "f64"  "f64"  neon           asimd     v               chained     r"fabs v\v\().2d, v\v\().2d"                                          "vabsq_f64";
    FCVTNU    "v.2d, v.2d"                                 "f64"  "u64"  neon           asimd     v               chained     r"fcvtnu v\v\().2d, v\v\().2d"                                        "vcvtnq_u64_f64";
    FNEG      "v.4s, v.4s"                                 "f32"  "f32"  neon           asimd     v               chained     r"fneg v\v\().4s, v\v\().4s"                                          "vnegq_f32";
    FRECPS    "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"frecps v\v\().4s, v\v\().4s, v\fixed\().4s"                         "vrecpsq_f32";
    LD1       "{v.4s, v.4s, v.4s, v.4s}, [x]"              "u32"  "u32"  neon           asimd     lines           throughput  r"ld1 {{v16.4s, v17.4s, v18.4s, v19.4s}}, [x\v]"                      "vld1q_u32_x4";
    MOV       "v.b[imm], w"                                "u8"   "u8"   neon           asimd     v               chained     r"mov v\v\().b[1], w8"                                                "vsetq_lane_u8";
    ORN       "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"orn v\v\().16b, v\v\().16b, v\fixed\().16b"                         "vornq_u32 vornq_u64 vornq_u8";
    SMAXV     "s, v.4s"                                    "i32"  "i32"  neon           asimd     v               chained     r"smaxv s\v, v\v\().4s"                                               "vmaxvq_s32";
    SSHLL2    "v.2d, v.4s, #imm"                           "i32"  "i64"  neon           asimd     v               chained     r"sshll2 v\v\().2d, v\v\().4s, #0"                                    "vmovl_high_s32";
    UMAX      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"umax v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vmaxq_u32";
    UMAXV     "h, v.8h"                                    "u16"  "u16"  neon           asimd     v               chained     r"umaxv h\v, v\v\().8h"                                               "vmaxvq_u16";
    UMIN      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"umin v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vminq_u16";
    UMULL     "v.4s, v.4h, v.4h"                           "u16"  "u32"  neon           asimd     v               chained     r"umull v\v\().4s, v\v\().4h, v\fixed\().4h"                          "vmull_u16";
    UMULL2    "v.4s, v.8h, v.8h"                           "u16"  "u32"  neon           asimd     v               chained     r"umull2 v\v\().4s, v\v\().8h, v\fixed\().8h"                         "vmull_high_u16";
    UMULL2    "v.8h, v.16b, v.16b"                         "u8"   "u16"  neon           asimd     v               chained     r"umull2 v\v\().8h, v\v\().16b, v\fixed\().16b"                       "vmull_high_u8";
    USHLL2    "v.2d, v.4s, #imm"                           "u32"  "u64"  neon           asimd     v               chained     r"ushll2 v\v\().2d, v\v\().4s, #0"                                    "vmovl_high_u32";
    UZP1      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"uzp1 v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vuzp1q_u32";
    UZP2      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"uzp2 v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vuzp2q_u32";
    AND       "v.8b, v.8b, v.8b"                           "u8"   "u8"   neon           asimd     v               chained     r"and v\v\().8b, v\v\().8b, v\fixed\().8b"                            "vand_u16 vand_u8";
    CMGE      "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"cmge v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vcgeq_s8";
    FCMGE     "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fcmge v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vcgeq_f32";
    FMAXNMV   "s, v.4s"                                    "f32"  "f32"  neon           asimd     v               chained     r"fmaxnmv s\v, v\v\().4s"                                             "vmaxnmvq_f32";
    FMINV     "s, v.4s"                                    "f32"  "f32"  neon           asimd     v               chained     r"fminv s\v, v\v\().4s"                                               "vminvq_f32";
    FMLA      "v.2s, v.2s, v.2s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmla v\v\().2s, v\fixed\().2s, v\fixed\().2s"                       "vfma_f32";
    FMULX     "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fmulx v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vmulxq_f64";
    FRSQRTS   "d, d, d"                                    "f64"  "f64"  neon           asimd     v               chained     r"frsqrts d\v, d\v, d\fixed"                                          "vrsqrtsd_f64";
    FSUB      "v.2s, v.2s, v.2s"                           "f32"  "f32"  neon           asimd     v               chained     r"fsub v\v\().2s, v\v\().2s, v\fixed\().2s"                           "vsub_f32";
    LD1       "{v.d}[imm], [x]"                            "u64"  "u64"  neon           asimd     lines           throughput  r"ld1 {{v\v\().d}}[1], [x\v]"                                         "vld1q_lane_f64";
    LD1       "{v.h}[imm], [x]"                            "u16"  "u16"  neon           asimd     lines           throughput  r"ld1 {{v\v\().h}}[1], [x\v]"                                         "vld1_lane_u16";
    LD2       "{v.8b, v.8b}, [x]"                          "u8"   "u8"   neon           asimd     lines           throughput  r"ld2 {{v16.8b, v17.8b}}, [x\v]"                                      "vld2_u8";
    LD2       "{v.h, v.h}[imm], [x]"                       "u16"  "u16"  neon           asimd     lines           throughput  r"ld2 {{v16.h, v17.h}}[1], [x\v]"                                     "vld2_lane_s16";
    LD3       "{v.8b, v.8b, v.8b}, [x]"                    "u8"   "u8"   neon           asimd     lines           throughput  r"ld3 {{v16.8b, v17.8b, v18.8b}}, [x\v]"                              "vld3_u8";
    LD3       "{v.h, v.h, v.h}[imm], [x]"                  "u16"  "u16"  neon           asimd     lines           throughput  r"ld3 {{v16.h, v17.h, v18.h}}[1], [x\v]"                              "vld3_lane_u16";
    LD3       "{v.s, v.s, v.s}[imm], [x]"                  "u32"  "u32"  neon           asimd     lines           throughput  r"ld3 {{v16.s, v17.s, v18.s}}[1], [x\v]"                              "vld3q_lane_s32";
    LD4       "{v.8b, v.8b, v.8b, v.8b}, [x]"              "u8"   "u8"   neon           asimd     lines           throughput  r"ld4 {{v16.8b, v17.8b, v18.8b, v19.8b}}, [x\v]"                      "vld4_u8";
    MOV       "v.d[imm], x"                                "u64"  "u64"  neon           asimd     v               chained     r"mov v\v\().d[1], x8"                                                "vsetq_lane_u64";
    NEG       "v.4s, v.4s"                                 "i32"  "i32"  neon           asimd     v               chained     r"neg v\v\().4s, v\v\().4s"                                           "vnegq_s32";
    REV64     "v.4s, v.4s"                                 "u32"  "u32"  neon           asimd     v               chained     r"rev64 v\v\().4s, v\v\().4s"                                         "vrev64q_u32";
    SABD      "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"sabd v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vabdq_s8";
    SMAXV     "h, v.8h"                                    "i16"  "i16"  neon           asimd     v               chained     r"smaxv h\v, v\v\().8h"                                               "vmaxvq_s16";
    SMINV     "h, v.8h"                                    "i16"  "i16"  neon           asimd     v               chained     r"sminv h\v, v\v\().8h"                                               "vminvq_s16";
    ST2       "{v.2d, v.2d}, [x]"                          "u64"  "u64"  neon           asimd     lines           throughput  r"st2 {{v16.2d, v17.2d}}, [x\v]"                                      "vst2q_f64";
    ST2       "{v.4s, v.4s}, [x]"                          "u32"  "u32"  neon           asimd     lines           throughput  r"st2 {{v16.4s, v17.4s}}, [x\v]"                                      "vst2q_f32";
    SUB       "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"sub v\v\().8h, v\v\().8h, v\fixed\().8h"                            "vsubq_s16";
    UABD      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"uabd v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vabdq_u8";
    UMAX      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"umax v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vmaxq_u8";
    ADD       "v.8b, v.8b, v.8b"                           "i8"   "i8"   neon           asimd     v               chained     r"add v\v\().8b, v\v\().8b, v\fixed\().8b"                            "vadd_u8 vshl_n_u8";
    ADDV      "b, v.16b"                                   "i8"   "i8"   neon           asimd     v               chained     r"addv b\v, v\v\().16b"                                               "vaddvq_u8";
    CMEQ      "v.4s, v.4s, #imm"                           "i32"  "i32"  neon           asimd     v               chained     r"cmeq v\v\().4s, v\v\().4s, #0"                                      "vceqzq_u32";
    CMHS      "v.2d, v.2d, v.2d"                           "u64"  "u64"  neon           asimd     v               chained     r"cmhs v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vcleq_u64";
    FADDP     "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"faddp v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vpaddq_f64";
    FCVTZS    "v.4s, v.4s"                                 "f32"  "i32"  neon           asimd     v               chained     r"fcvtzs v\v\().4s, v\v\().4s"                                        "vcvtq_s32_f32";
    FMULX     "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"fmulx v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vmulxq_f32";
    FRECPE    "v.4s, v.4s"                                 "f32"  "f32"  neon           asimd     v               chained     r"frecpe v\v\().4s, v\v\().4s"                                        "vrecpeq_f32";
    FRINTN    "v.2d, v.2d"                                 "f64"  "f64"  neon           asimd     v               chained     r"frintn v\v\().2d, v\v\().2d"                                        "vrndnq_f64";
    FRSQRTE   "v.2d, v.2d"                                 "f64"  "f64"  neon           asimd     v               chained     r"frsqrte v\v\().2d, v\v\().2d"                                       "vrsqrteq_f64";
    FRSQRTS   "s, s, s"                                    "f32"  "f32"  neon           asimd     v               chained     r"frsqrts s\v, s\v, s\fixed"                                          "vrsqrtss_f32";
    FRSQRTS   "v.2s, v.2s, v.2s"                           "f32"  "f32"  neon           asimd     v               chained     r"frsqrts v\v\().2s, v\v\().2s, v\fixed\().2s"                        "vrsqrts_f32";
    FRSQRTS   "v.4s, v.4s, v.4s"                           "f32"  "f32"  neon           asimd     v               chained     r"frsqrts v\v\().4s, v\v\().4s, v\fixed\().4s"                        "vrsqrtsq_f32";
    LD1R      "{v.4s}, [x]"                                "u32"  "u32"  neon           asimd     lines           throughput  r"ld1r {{v\v\().4s}}, [x\v]"                                          "vld1q_dup_f32";
    LD1R      "{v.8b}, [x]"                                "u8"   "u8"   neon           asimd     lines           throughput  r"ld1r {{v\v\().8b}}, [x\v]"                                          "vld1_dup_u8";
    ORR       "v.8b, v.8b, v.8b"                           "u8"   "u8"   neon           asimd     v               chained     r"orr v\v\().8b, v\v\().8b, v\fixed\().8b"                            "vorr_u8";
    SADALP    "v.2d, v.4s"                                 "i32"  "i64"  neon           asimd     v               chained     r"sadalp v\v\().2d, v\fixed\().4s"                                    "vpadalq_s32";
    SHRN      "v.4h, v.4s, #imm"                           "i32"  "i16"  neon           asimd     v               chained     r"shrn v\v\().4h, v\v\().4s, #1"                                      "vshrn_n_u32";
    SMAXV     "b, v.16b"                                   "i8"   "i8"   neon           asimd     v               chained     r"smaxv b\v, v\v\().16b"                                              "vmaxvq_s8";
    SMINV     "b, v.16b"                                   "i8"   "i8"   neon           asimd     v               chained     r"sminv b\v, v\v\().16b"                                              "vminvq_s8";
    SMINV     "s, v.4s"                                    "i32"  "i32"  neon           asimd     v               chained     r"sminv s\v, v\v\().4s"                                               "vminvq_s32";
    SMLAL     "v.2d, v.2s, v.2s"                           "i32"  "i64"  neon           asimd     v               chained     r"smlal v\v\().2d, v\fixed\().2s, v\fixed\().2s"                      "vmlal_s32";
    SMLAL     "v.8h, v.8b, v.8b"                           "i8"   "i16"  neon           asimd     v               chained     r"smlal v\v\().8h, v\fixed\().8b, v\fixed\().8b"                      "vmlal_s8";
    SMLAL2    "v.2d, v.4s, v.4s"                           "i32"  "i64"  neon           asimd     v               chained     r"smlal2 v\v\().2d, v\fixed\().4s, v\fixed\().4s"                     "vmlal_high_s32";
    SMLAL2    "v.8h, v.16b, v.16b"                         "i8"   "i16"  neon           asimd     v               chained     r"smlal2 v\v\().8h, v\fixed\().16b, v\fixed\().16b"                   "vmlal_high_s8";
    SQADD     "v.16b, v.16b, v.16b"                        "i8"   "i8"   neon           asimd     v               chained     r"sqadd v\v\().16b, v\v\().16b, v\fixed\().16b"                       "vqaddq_s8";
    SQXTN     "v.2s, v.2d"                                 "i64"  "i32"  neon           asimd     v               chained     r"sqxtn v\v\().2s, v\v\().2d"                                         "vqmovn_s64";
    SQXTN     "v.8b, v.8h"                                 "i16"  "i8"   neon           asimd     v               chained     r"sqxtn v\v\().8b, v\v\().8h"                                         "vqmovn_s16";
    SSHL      "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"sshl v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vshlq_s32";
    UADALP    "v.2d, v.4s"                                 "u32"  "u64"  neon           asimd     v               chained     r"uadalp v\v\().2d, v\fixed\().4s"                                    "vpadalq_u32";
    UADDLV    "d, v.4s"                                    "u32"  "u64"  neon           asimd     v               chained     r"uaddlv d\v, v\v\().4s"                                              "vaddlvq_u32";
    UADDW     "v.2d, v.2d, v.2s"                           "u32"  "u64"  neon           asimd     v               chained     r"uaddw v\v\().2d, v\v\().2d, v\fixed\().2s"                          "vaddw_u32";
    UADDW2    "v.2d, v.2d, v.4s"                           "u32"  "u64"  neon           asimd     v               chained     r"uaddw2 v\v\().2d, v\v\().2d, v\fixed\().4s"                         "vaddw_high_u32";
    UQADD     "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"uqadd v\v\().16b, v\v\().16b, v\fixed\().16b"                       "vqaddq_u8";
    UQXTN     "v.2s, v.2d"                                 "u64"  "u32"  neon           asimd     v               chained     r"uqxtn v\v\().2s, v\v\().2d"                                         "vqmovn_u64";
    UQXTN     "v.8b, v.8h"                                 "u16"  "u8"   neon           asimd     v               chained     r"uqxtn v\v\().8b, v\v\().8h"                                         "vqmovn_u16";
    USHL      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"ushl v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vshlq_u8";
    USHL      "v.2d, v.2d, v.2d"                           "u64"  "u64"  neon           asimd     v               chained     r"ushl v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vshlq_u64";
    USHR      "v.8b, v.8b, #imm"                           "u8"   "u8"   neon           asimd     v               chained     r"ushr v\v\().8b, v\v\().8b, #1"                                      "vshr_n_u8";
    SMLSL     "v.2d, v.2s, v.2s"                           "i32"  "i64"  neon           asimd     v               chained     r"smlsl v\v\().2d, v\fixed\().2s, v\fixed\().2s"                      "vmlsl_n_s32 vmlsl_s32";
    UMLSL     "v.2d, v.2s, v.2s"                           "u32"  "u64"  neon           asimd     v               chained     r"umlsl v\v\().2d, v\fixed\().2s, v\fixed\().2s"                      "vmlsl_n_u32 vmlsl_u32";
    SMLSL2    "v.2d, v.4s, v.4s"                           "i32"  "i64"  neon           asimd     v               chained     r"smlsl2 v\v\().2d, v\fixed\().4s, v\fixed\().4s"                     "vmlsl_high_n_s32 vmlsl_high_s32";
    UMLSL2    "v.2d, v.4s, v.4s"                           "u32"  "u64"  neon           asimd     v               chained     r"umlsl2 v\v\().2d, v\fixed\().4s, v\fixed\().4s"                     "vmlsl_high_n_u32 vmlsl_high_u32";
    ZIP1      "v.8b, v.8b, v.8b"                           "u8"   "u8"   neon           asimd     v               chained     r"zip1 v\v\().8b, v\v\().8b, v\fixed\().8b"                           "vzip1_u8 vzip_u8";
    ABS       "v.2d, v.2d"                                 "i64"  "i64"  neon           asimd     v               chained     r"abs v\v\().2d, v\v\().2d"                                           "vabsq_s64";
    ADDV      "h, v.4h"                                    "i16"  "i16"  neon           asimd     v               chained     r"addv h\v, v\v\().4h"                                                "vaddv_u16";
    CMGE      "v.2d, v.2d, v.2d"                           "i64"  "i64"  neon           asimd     v               chained     r"cmge v\v\().2d, v\v\().2d, v\fixed\().2d"                           "vcleq_s64";
    CMGT      "v.4s, v.4s, #imm"                           "i32"  "i32"  neon           asimd     v               chained     r"cmgt v\v\().4s, v\v\().4s, #0"                                      "vcgtzq_s32";
    CMTST     "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"cmtst v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vtstq_u32";
    CMTST     "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"cmtst v\v\().8h, v\v\().8h, v\fixed\().8h"                          "vtstq_u16";
    CNT       "v.8b, v.8b"                                 "u8"   "u8"   neon           asimd     v               chained     r"cnt v\v\().8b, v\v\().8b"                                           "vcnt_u8";
    EXT       "v.8b, v.8b, v.8b, #imm"                     "u8"   "u8"   neon           asimd     v               chained     r"ext v\v\().8b, v\v\().8b, v\fixed\().8b, #1"                        "vext_u8";
    FCMGE     "v.4s, v.4s, #imm"                           "f32"  "f32"  neon           asimd     v               chained     r"fcmge v\v\().4s, v\v\().4s, #0.0"                                   "vcgezq_f32";
    FCVTZU    "v.2d, v.2d"                                 "f64"  "u64"  neon           asimd     v               chained     r"fcvtzu v\v\().2d, v\v\().2d"                                        "vcvtq_u64_f64";
    FMAXP     "d, v.2d"                                    "f64"  "f64"  neon           asimd     v               chained     r"fmaxp d\v, v\v\().2d"                                               "vmaxvq_f64";
    FMINP     "d, v.2d"                                    "f64"  "f64"  neon           asimd     v               chained     r"fminp d\v, v\v\().2d"                                               "vminvq_f64";
    FMINP     "v.2d, v.2d, v.2d"                           "f64"  "f64"  neon           asimd     v               chained     r"fminp v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vpminq_f64";
    SQADD     "d, d, d"                                    "i64"  "i64"  neon           asimd     v               chained     r"sqadd d\v, d\v, d\fixed"                                            "vqaddd_s64";
    UQADD     "d, d, d"                                    "u64"  "u64"  neon           asimd     v               chained     r"uqadd d\v, d\v, d\fixed"                                            "vqaddd_u64";
    SQADD     "s, s, s"                                    "i32"  "i32"  neon           asimd     v               chained     r"sqadd s\v, s\v, s\fixed"                                            "vqadds_s32";
    SQADD     "v.4h, v.4h, v.4h"                           "i16"  "i16"  neon           asimd     v               chained     r"sqadd v\v\().4h, v\v\().4h, v\fixed\().4h"                          "vqaddh_s16";
    SQADD     "v.8b, v.8b, v.8b"                           "i8"   "i8"   neon           asimd     v               chained     r"sqadd v\v\().8b, v\v\().8b, v\fixed\().8b"                          "vqaddb_s8";
    UQADD     "s, s, s"                                    "u32"  "u32"  neon           asimd     v               chained     r"uqadd s\v, s\v, s\fixed"                                            "vqadds_u32";
    UQADD     "v.4h, v.4h, v.4h"                           "u16"  "u16"  neon           asimd     v               chained     r"uqadd v\v\().4h, v\v\().4h, v\fixed\().4h"                          "vqaddh_u16";
    UQADD     "v.8b, v.8b, v.8b"                           "u8"   "u8"   neon           asimd     v               chained     r"uqadd v\v\().8b, v\v\().8b, v\fixed\().8b"                          "vqaddb_u8";
    FRINTN    "v.4s, v.4s"                                 "f32"  "f32"  neon           asimd     v               chained     r"frintn v\v\().4s, v\v\().4s"                                        "vrndnq_f32";
    FRSQRTE   "d, d"                                       "f64"  "f64"  neon           asimd     v               chained     r"frsqrte d\v, d\v"                                                   "vrsqrted_f64";
    FRSQRTE   "s, s"                                       "f32"  "f32"  neon           asimd     v               chained     r"frsqrte s\v, s\v"                                                   "vrsqrtes_f32";
    FRSQRTE   "v.2s, v.2s"                                 "f32"  "f32"  neon           asimd     v               chained     r"frsqrte v\v\().2s, v\v\().2s"                                       "vrsqrte_f32";
    FRSQRTE   "v.4s, v.4s"                                 "f32"  "f32"  neon           asimd     v               chained     r"frsqrte v\v\().4s, v\v\().4s"                                       "vrsqrteq_f32";
    LD1       "{v.b}[imm], [x]"                            "u8"   "u8"   neon           asimd     lines           throughput  r"ld1 {{v\v\().b}}[1], [x\v]"                                         "vld1_lane_u8";
    LD1       "{v.s}[imm], [x]"                            "u32"  "u32"  neon           asimd     lines           throughput  r"ld1 {{v\v\().s}}[1], [x\v]"                                         "vld1q_lane_u32";
    LD3       "{v.2d, v.2d, v.2d}, [x]"                    "u64"  "u64"  neon           asimd     lines           throughput  r"ld3 {{v16.2d, v17.2d, v18.2d}}, [x\v]"                              "vld3q_f64";
    LD3       "{v.2s, v.2s, v.2s}, [x]"                    "u32"  "u32"  neon           asimd     lines           throughput  r"ld3 {{v16.2s, v17.2s, v18.2s}}, [x\v]"                              "vld3_f32";
    LD3       "{v.4h, v.4h, v.4h}, [x]"                    "u16"  "u16"  neon           asimd     lines           throughput  r"ld3 {{v16.4h, v17.4h, v18.4h}}, [x\v]"                              "vld3_u16";
    LD3       "{v.d, v.d, v.d}[imm], [x]"                  "u64"  "u64"  neon           asimd     lines           throughput  r"ld3 {{v16.d, v17.d, v18.d}}[1], [x\v]"                              "vld3q_lane_f64";
    MOV       "v.s[imm], v.s[imm]"                         "u32"  "u32"  neon           asimd     v               chained     r"mov v\v\().s[1], v\fixed\().s[1]"                                   "vsetq_lane_f32";
    MOV       "v.s[imm], w"                                "u32"  "u32"  neon           asimd     v               chained     r"mov v\v\().s[1], w8"                                                "vsetq_lane_u32";
    REV64     "v.16b, v.16b"                               "u8"   "u8"   neon           asimd     v               chained     r"rev64 v\v\().16b, v\v\().16b"                                       "vrev64q_u8";
    SLI       "v.16b, v.16b, #imm"                         "u8"   "u8"   neon           asimd     v               chained     r"sli v\v\().16b, v\fixed\().16b, #1"                                 "vsliq_n_u8";
    SMAX      "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"smax v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vmaxq_s16";
    SMIN      "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"smin v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vminq_s16";
    SQADD     "v.2d, v.2d, v.2d"                           "i64"  "i64"  neon           asimd     v               chained     r"sqadd v\v\().2d, v\v\().2d, v\fixed\().2d"                          "vqaddq_s64";
    SQADD     "v.4s, v.4s, v.4s"                           "i32"  "i32"  neon           asimd     v               chained     r"sqadd v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vqaddq_s32";
    SQADD     "v.8h, v.8h, v.8h"                           "i16"  "i16"  neon           asimd     v               chained     r"sqadd v\v\().8h, v\v\().8h, v\fixed\().8h"                          "vqaddq_s16";
    SSHLL2    "v.8h, v.16b, #imm"                          "i8"   "i16"  neon           asimd     v               chained     r"sshll2 v\v\().8h, v\v\().16b, #0"                                   "vmovl_high_s8";
    SSHR      "v.8h, v.8h, #imm"                           "i16"  "i16"  neon           asimd     v               chained     r"sshr v\v\().8h, v\v\().8h, #1"                                      "vshrq_n_s16";
    ST1       "{v.b}[imm], [x]"                            "u8"   "u8"   neon           asimd     lines           throughput  r"st1 {{v\v\().b}}[1], [x\v]"                                         "vst1_lane_u8";
    ST2       "{v.16b, v.16b}, [x]"                        "u8"   "u8"   neon           asimd     lines           throughput  r"st2 {{v16.16b, v17.16b}}, [x\v]"                                    "vst2q_u8";
    ST4       "{v.16b, v.16b, v.16b, v.16b}, [x]"          "u8"   "u8"   neon           asimd     lines           throughput  r"st4 {{v16.16b, v17.16b, v18.16b, v19.16b}}, [x\v]"                  "vst4q_u8";
    SUB       "v.2s, v.2s, v.2s"                           "i32"  "i32"  neon           asimd     v               chained     r"sub v\v\().2s, v\v\().2s, v\fixed\().2s"                            "vsub_u32";
    TBL       "v.8b, {v.16b, v.16b}, v.8b"                 "u8"   "u8"   neon           asimd     v_low           chained     r"tbl v\v\().8b, {{v16.16b, v17.16b}}, v\v\().8b"                     "vqtbl2_u8";
    TRN1      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"trn1 v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vtrn1q_u8";
    TRN1      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"trn1 v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vtrn1q_u32";
    TRN1      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"trn1 v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vtrn1q_u16";
    TRN2      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"trn2 v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vtrn2q_u8";
    TRN2      "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"trn2 v\v\().4s, v\v\().4s, v\fixed\().4s"                           "vtrn2q_u32";
    TRN2      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"trn2 v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vtrn2q_u16";
    UADDLV    "s, v.8h"                                    "u16"  "u32"  neon           asimd     v               chained     r"uaddlv s\v, v\v\().8h"                                              "vaddlvq_u16";
    UQADD     "v.4s, v.4s, v.4s"                           "u32"  "u32"  neon           asimd     v               chained     r"uqadd v\v\().4s, v\v\().4s, v\fixed\().4s"                          "vqaddq_u32";
    UQSUB     "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"uqsub v\v\().8h, v\v\().8h, v\fixed\().8h"                          "vqsubq_u16";
    USHL      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"ushl v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vshlq_u16";
    USHR      "v.2s, v.2s, #imm"                           "u32"  "u32"  neon           asimd     v               chained     r"ushr v\v\().2s, v\v\().2s, #1"                                      "vshr_n_u32";
    UZP1      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"uzp1 v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vuzp1q_u8";
    UZP1      "v.8b, v.8b, v.8b"                           "u8"   "u8"   neon           asimd     v               chained     r"uzp1 v\v\().8b, v\v\().8b, v\fixed\().8b"                           "vuzp1_u8";
    UZP1      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"uzp1 v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vuzp1q_u16";
    UZP2      "v.16b, v.16b, v.16b"                        "u8"   "u8"   neon           asimd     v               chained     r"uzp2 v\v\().16b, v\v\().16b, v\fixed\().16b"                        "vuzp2q_u8";
    UZP2      "v.8h, v.8h, v.8h"                           "u16"  "u16"  neon           asimd     v               chained     r"uzp2 v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vuzp2q_u16";
    ZIP2      "v.8b, v.8b, v.8b"                           "u8"   "u8"   neon           asimd     v               chained     r"zip2 v\v\().8b, v\v\().8b, v\fixed\().8b"                           "vzip_u8";

    FMLA      "v.8h, v.8h, v.8h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fmla v\v\().8h, v\fixed\().8h, v\fixed\().8h"                       "vfmaq_f16";
    FMUL      "v.8h, v.8h, v.8h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fmul v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vmulq_f16";
    FADD      "v.8h, v.8h, v.8h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fadd v\v\().8h, v\v\().8h, v\fixed\().8h"                           "vaddq_f16";
    FMUL      "v.4h, v.4h, v.4h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fmul v\v\().4h, v\v\().4h, v\fixed\().4h"                           "vmul_f16";
    FACGT     "v.8h, v.8h, v.8h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"facgt v\v\().8h, v\v\().8h, v\fixed\().8h"                          "vcagtq_f16";
    FMLS      "v.8h, v.8h, v.8h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fmls v\v\().8h, v\fixed\().8h, v\fixed\().8h"                       "vfmsq_f16";
    FCVTZS    "v.8h, v.8h"                                 "f16"  "i16"  fp16           asimdhp   v               chained     r"fcvtzs v\v\().8h, v\v\().8h"                                        "vcvtq_s16_f16";
    FRINTN    "v.8h, v.8h"                                 "f16"  "f16"  fp16           asimdhp   v               chained     r"frintn v\v\().8h, v\v\().8h"                                        "vrndnq_f16";
    FRSQRTS   "v.4h, v.4h, v.4h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"frsqrts v\v\().4h, v\v\().4h, v\fixed\().4h"                        "vrsqrts_f16";
    FABS      "v.8h, v.8h"                                 "f16"  "f16"  fp16           asimdhp   v               chained     r"fabs v\v\().8h, v\v\().8h"                                          "vabsq_f16";
    FDIV      "v.8h, v.8h, v.8h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fdiv v\v\().8h, v\v\().8h, v\v\().8h"                               "vdivq_f16";
    FMLA      "v.4h, v.4h, v.4h"                           "f16"  "f16"  fp16           asimdhp   v               chained     r"fmla v\v\().4h, v\fixed\().4h, v\fixed\().4h"                       "vfma_f16";
    FNEG      "v.8h, v.8h"                                 "f16"  "f16"  fp16           asimdhp   v               chained     r"fneg v\v\().8h, v\v\().8h"                                          "vnegq_f16";
    FRSQRTE   "v.4h, v.4h"                                 "f16"  "f16"  fp16           asimdhp   v               chained     r"frsqrte v\v\().4h, v\v\().4h"                                       "vrsqrte_f16";
    FSQRT     "v.4h, v.4h"                                 "f16"  "f16"  fp16           asimdhp   v               chained     r"fsqrt v\v\().4h, v\v\().4h"                                         "vsqrt_f16";

    FMLAL     "v.4s, v.4h, v.4h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlal v\v\().4s, v\fixed\().4h, v\fixed\().4h"                      "vfmlalq_low_f16";
    FMLAL2    "v.4s, v.4h, v.4h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlal2 v\v\().4s, v\fixed\().4h, v\fixed\().4h"                     "vfmlalq_high_f16";
    FMLAL     "v.2s, v.2h, v.2h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlal v\v\().2s, v\fixed\().2h, v\fixed\().2h"                      "vfmlal_low_f16";
    FMLAL2    "v.2s, v.2h, v.2h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlal2 v\v\().2s, v\fixed\().2h, v\fixed\().2h"                     "vfmlal_high_f16";
    FMLSL     "v.4s, v.4h, v.4h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlsl v\v\().4s, v\fixed\().4h, v\fixed\().4h"                      "vfmlslq_low_f16";
    FMLSL2    "v.4s, v.4h, v.4h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlsl2 v\v\().4s, v\fixed\().4h, v\fixed\().4h"                     "vfmlslq_high_f16";
    FMLSL     "v.2s, v.2h, v.2h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlsl v\v\().2s, v\fixed\().2h, v\fixed\().2h"                      "vfmlsl_low_f16";
    FMLSL2    "v.2s, v.2h, v.2h"                           "f16"  "f32"  fp16fml        asimdfhm  v               chained     r"fmlsl2 v\v\().2s, v\fixed\().2h, v\fixed\().2h"                     "vfmlsl_high_f16";

    SHLL      "v.4s, v.4h, #imm"                           "bf16" "f32"  bf16           asimd     v               chained     r"shll v\v\().4s, v\v\().4h, #16"                                     "vcvt_f32_bf16";
    BFDOT     "v.4s, v.8h, v.8h"                           "bf16" "f32"  bf16           bf16      v               chained     r"bfdot v\v\().4s, v\fixed\().8h, v\fixed\().8h"                      "vbfdotq_f32";
    BFMLALB   "v.4s, v.8h, v.8h"                           "bf16" "f32"  bf16           bf16      v               chained     r"bfmlalb v\v\().4s, v\fixed\().8h, v\fixed\().8h"                    "vbfmlalbq_f32";
    BFMLALT   "v.4s, v.8h, v.8h"                           "bf16" "f32"  bf16           bf16      v               chained     r"bfmlalt v\v\().4s, v\fixed\().8h, v\fixed\().8h"                    "vbfmlaltq_f32";
    BFCVTN    "v.4h, v.4s"                                 "f32"  "bf16" bf16           bf16      v               chained     r"bfcvtn v\v\().4h, v\v\().4s"                                        "vcvt_bf16_f32";
    BFMMLA    "v.4s, v.8h, v.8h"                           "bf16" "f32"  bf16           bf16      v               chained     r"bfmmla v\v\().4s, v\fixed\().8h, v\fixed\().8h"                     "";

    SDOT      "v.4s, v.16b, v.16b"                         "i8"   "i32"  dotprod        asimddp   v               chained     r"sdot v\v\().4s, v\fixed\().16b, v\fixed\().16b"                     "vdotq_s32";
    UDOT      "v.4s, v.16b, v.16b"                         "u8"   "u32"  dotprod        asimddp   v               chained     r"udot v\v\().4s, v\fixed\().16b, v\fixed\().16b"                     "vdotq_u32";

    SMMLA     "v.4s, v.16b, v.16b"                         "i8"   "i32"  i8mm           i8mm      v               chained     r"smmla v\v\().4s, v\fixed\().16b, v\fixed\().16b"                    "";

    AESE      "v.16b, v.16b"                               "u8"   "u8"   aes            aes       v               chained     r"aese v\v\().16b, v\fixed\().16b"                                    "vaeseq_u8";
    AESMC     "v.16b, v.16b"                               "u8"   "u8"   aes            aes       v               chained     r"aesmc v\v\().16b, v\v\().16b"                                       "vaesmcq_u8";
    PMULL     "v.1q, v.1d, v.1d"                           "p64"  "p128" aes            pmull     v               chained     r"pmull v\v\().1q, v\v\().1d, v\fixed\().1d"                          "vmull_p64";
    PMULL2    "v.1q, v.2d, v.2d"                           "p64"  "p128" aes            pmull     v               chained     r"pmull2 v\v\().1q, v\v\().2d, v\fixed\().2d"                         "vmull_high_p64";

    SHA256H   "q, q, v.4s"                                 "u32"  "u32"  sha2           sha2      v               chained     r"sha256h q\v, q\fixed, v\fixed\().4s"                                "vsha256hq_u32";
    SHA256H2  "q, q, v.4s"                                 "u32"  "u32"  sha2           sha2      v               chained     r"sha256h2 q\v, q\fixed, v\fixed\().4s"                               "vsha256h2q_u32";
    SHA256SU0 "v.4s, v.4s"                                 "u32"  "u32"  sha2           sha2      v               chained     r"sha256su0 v\v\().4s, v\fixed\().4s"                                 "vsha256su0q_u32";
    SHA256SU1 "v.4s, v.4s, v.4s"                           "u32"  "u32"  sha2           sha2      v               chained     r"sha256su1 v\v\().4s, v\fixed\().4s, v\fixed\().4s"                  "vsha256su1q_u32";

    CLZ       "x, x"                                       "u64"  "u64"  base           fp        x               chained     r"clz x\v, x\v"                                                       "__clzll";
    PRFM      "pldl2keep, [x]"                             ""     ""     base           fp        lines           throughput  r"prfm pldl2keep, [x\v]"                                              "";
    DC        "cvac, x"                                    ""     ""     base           fp        lines           throughput  r"dc cvac, x\v"                                                       "";
    LDR       "x, [x]"                                     "u64"  "u64"  base           fp        lines           chained     r"ldr x\v, [x\v]"                                                     "";
    PRFM      "pstl1keep, [x]"                             ""     ""     base           fp        lines           throughput  r"prfm pstl1keep, [x\v]"                                              "";
    STR       "x, [x]"                                     "u64"  "u64"  base           fp        lines           throughput  r"str x\v, [x\v]"                                                     "";
    UMULH     "x, x, x"                                    "u64"  "u64"  base           fp        x               chained     r"umulh x\v, x\v, x\fixed"                                            "";
    FMADD     "d, d, d, d"                                 "f64"  "f64"  base           fp        v               chained     r"fmadd d\v, d\fixed, d\fixed, d\v"                                   "vfma_f64";
    CLZ       "w, w"                                       "u32"  "u32"  base           fp        x               chained     r"clz w\v, w\v"                                                       "__clz";
    RBIT      "x, x"                                       "u64"  "u64"  base           fp        x               chained     r"rbit x\v, x\v"                                                      "__rbitll";
    CLREX     ""                                           ""     ""     base           fp        x               throughput  r"clrex"                                                              "";
    FMADD     "s, s, s, s"                                 "f32"  "f32"  base           fp        v               chained     r"fmadd s\v, s\fixed, s\fixed, s\v"                                   "";
    LDAXRB    "w, [x]"                                     "u8"   "u8"   base           fp        lines           throughput  r"ldaxrb w1\v, [x\v]"                                                 "";
    LDAXRH    "w, [x]"                                     "u16"  "u16"  base           fp        lines           throughput  r"ldaxrh w1\v, [x\v]"                                                 "";
    LDRB      "w, [x]"                                     "u8"   "u8"   base           fp        lines           throughput  r"ldrb w1\v, [x\v]"                                                   "";
    LDRH      "w, [x]"                                     "u16"  "u16"  base           fp        lines           throughput  r"ldrh w1\v, [x\v]"                                                   "";
    MRS       "x, CNTFRQ_EL0"                              "u64"  "u64"  base           fp        x               throughput  r"mrs x\v, CNTFRQ_EL0"                                                "";
    MRS       "x, CNTVCT_EL0"                              "u64"  "u64"  base           fp        x               throughput  r"mrs x\v, CNTVCT_EL0"                                                "";
    MRS       "x, fpcr"                                    "u64"  "u64"  base           fp        x               throughput  r"mrs x\v, fpcr"                                                      "";
    MSR       "fpcr, x"                                    ""     ""     base           fp        x               throughput  r"msr fpcr, xzr"                                                      "";
    STRB      "w, [x]"                                     "u8"   "u8"   base           fp        lines           throughput  r"strb w1\v, [x\v]"                                                   "";
    STRH      "w, [x]"                                     "u16"  "u16"  base           fp        lines           throughput  r"strh w1\v, [x\v]"                                                   "";
    RBIT      "w, w"                                       "u32"  "u32"  base           fp        x               chained     r"rbit w\v, w\v"                                                      "__rbit";

    YIELD     ""                                           ""     ""     atomics        fp        x               throughput  r"yield"                                                              "__yield";
    LDAR      "w, [x]"                                     "u32"  "u32"  atomics        fp        lines           throughput  r"ldar w1\v, [x\v]"                                                   "";
    LDAR      "x, [x]"                                     "u64"  "u64"  atomics        fp        lines           chained     r"ldar x\v, [x\v]"                                                    "";
    STLR      "w, [x]"                                     "u32"  "u32"  atomics        fp        lines           throughput  r"stlr w1\v, [x\v]"                                                   "";
    STLR      "x, [x]"                                     "u64"  "u64"  atomics        fp        lines           throughput  r"stlr x\v, [x\v]"                                                    "";
    LDAXR     "w, [x]"                                     "u32"  "u32"  atomics        fp        lines           throughput  r"ldaxr w1\v, [x\v]"                                                  "";
    LDAXR     "x, [x]"                                     "u64"  "u64"  atomics        fp        lines           chained     r"ldaxr x\v, [x\v]"                                                   "";
    DMB       "ish"                                        ""     ""     atomics        fp        x               throughput  r"dmb ish"                                                            "";
    LDXR      "w, [x]"                                     "u32"  "u32"  atomics        fp        lines           throughput  r"ldxr w1\v, [x\v]"                                                   "";
    LDXR      "x, [x]"                                     "u64"  "u64"  atomics        fp        lines           chained     r"ldxr x\v, [x\v]"                                                    "";
    SEVL      ""                                           ""     ""     atomics        fp        x               throughput  r"sevl"                                                               "";

    SWP       "w, w, [x]"                                  "u32"  "u32"  lse            atomics   lines           chained     r"swp w1\v, w1\v, [x\v]"                                              "";
    SWP       "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"swp x\v, x\v, [x\v]"                                                "";
    CAS       "w, w, [x]"                                  "u32"  "u32"  lse            atomics   lines           chained     r"cas w1\v, w1\v, [x\v]"                                              "";
    CAS       "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"cas x\v, x\v, [x\v]"                                                "";
    CASAL     "w, w, [x]"                                  "u32"  "u32"  lse            atomics   lines           chained     r"casal w1\v, w1\v, [x\v]"                                            "";
    CASAL     "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"casal x\v, x\v, [x\v]"                                              "";
    LDADD     "w, w, [x]"                                  "i32"  "i32"  lse            atomics   lines           chained     r"ldadd w1\v, w1\v, [x\v]"                                            "";
    LDADD     "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldadd xzr, x\v, [x\v]"                                              "";
    LDADDAL   "w, w, [x]"                                  "i32"  "i32"  lse            atomics   lines           chained     r"ldaddal w1\v, w1\v, [x\v]"                                          "";
    LDADDAL   "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldaddal xzr, x\v, [x\v]"                                            "";
    LDCLR     "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldclr xzr, x\v, [x\v]"                                              "";
    LDCLRAL   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldclral xzr, x\v, [x\v]"                                            "";
    LDSET     "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldset xzr, x\v, [x\v]"                                              "";
    LDSETAL   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldsetal xzr, x\v, [x\v]"                                            "";
    SWPAL     "w, w, [x]"                                  "u32"  "u32"  lse            atomics   lines           chained     r"swpal w1\v, w1\v, [x\v]"                                            "";
    SWPAL     "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"swpal x\v, x\v, [x\v]"                                              "";
    CASA      "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"casa x\v, x\v, [x\v]"                                               "";
    CASL      "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"casl x\v, x\v, [x\v]"                                               "";
    LDADDA    "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldadda xzr, x\v, [x\v]"                                             "";
    LDADDL    "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldaddl xzr, x\v, [x\v]"                                             "";
    LDCLRA    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldclra xzr, x\v, [x\v]"                                             "";
    LDCLRL    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldclrl xzr, x\v, [x\v]"                                             "";
    LDEOR     "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldeor xzr, x\v, [x\v]"                                              "";
    LDEORA    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldeora xzr, x\v, [x\v]"                                             "";
    LDEORAL   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldeoral xzr, x\v, [x\v]"                                            "";
    LDEORL    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldeorl xzr, x\v, [x\v]"                                             "";
    LDSETA    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldseta xzr, x\v, [x\v]"                                             "";
    LDSETL    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldsetl xzr, x\v, [x\v]"                                             "";
    LDSMAX    "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsmax x\v, x\v, [x\v]"                                             "";
    LDSMAXA   "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsmaxa x\v, x\v, [x\v]"                                            "";
    LDSMAXAL  "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsmaxal x\v, x\v, [x\v]"                                           "";
    LDSMAXL   "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsmaxl x\v, x\v, [x\v]"                                            "";
    LDSMIN    "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsmin x\v, x\v, [x\v]"                                             "";
    LDSMINA   "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsmina x\v, x\v, [x\v]"                                            "";
    LDSMINAL  "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsminal x\v, x\v, [x\v]"                                           "";
    LDSMINL   "x, x, [x]"                                  "i64"  "i64"  lse            atomics   lines           chained     r"ldsminl x\v, x\v, [x\v]"                                            "";
    LDUMAX    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldumax x\v, x\v, [x\v]"                                             "";
    LDUMAXA   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldumaxa x\v, x\v, [x\v]"                                            "";
    LDUMAXAL  "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldumaxal x\v, x\v, [x\v]"                                           "";
    LDUMAXL   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldumaxl x\v, x\v, [x\v]"                                            "";
    LDUMIN    "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldumin x\v, x\v, [x\v]"                                             "";
    LDUMINA   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"ldumina x\v, x\v, [x\v]"                                            "";
    LDUMINAL  "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"lduminal x\v, x\v, [x\v]"                                           "";
    LDUMINL   "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"lduminl x\v, x\v, [x\v]"                                            "";
    STADD     "x, [x]"                                     "i64"  "i64"  lse            atomics   lines           throughput  r"stadd x\v, [x\v]"                                                   "";
    STADDL    "x, [x]"                                     "i64"  "i64"  lse            atomics   lines           throughput  r"staddl x\v, [x\v]"                                                  "";
    STCLR     "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stclr x\v, [x\v]"                                                   "";
    STCLRL    "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stclrl x\v, [x\v]"                                                  "";
    STEOR     "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"steor x\v, [x\v]"                                                   "";
    STEORL    "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"steorl x\v, [x\v]"                                                  "";
    STSET     "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stset x\v, [x\v]"                                                   "";
    STSETL    "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stsetl x\v, [x\v]"                                                  "";
    STSMAX    "x, [x]"                                     "i64"  "i64"  lse            atomics   lines           throughput  r"stsmax x\v, [x\v]"                                                  "";
    STSMAXL   "x, [x]"                                     "i64"  "i64"  lse            atomics   lines           throughput  r"stsmaxl x\v, [x\v]"                                                 "";
    STSMIN    "x, [x]"                                     "i64"  "i64"  lse            atomics   lines           throughput  r"stsmin x\v, [x\v]"                                                  "";
    STSMINL   "x, [x]"                                     "i64"  "i64"  lse            atomics   lines           throughput  r"stsminl x\v, [x\v]"                                                 "";
    STUMAX    "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stumax x\v, [x\v]"                                                  "";
    STUMAXL   "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stumaxl x\v, [x\v]"                                                 "";
    STUMIN    "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stumin x\v, [x\v]"                                                  "";
    STUMINL   "x, [x]"                                     "u64"  "u64"  lse            atomics   lines           throughput  r"stuminl x\v, [x\v]"                                                 "";
    SWPA      "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"swpa x\v, x\v, [x\v]"                                               "";
    SWPL      "x, x, [x]"                                  "u64"  "u64"  lse            atomics   lines           chained     r"swpl x\v, x\v, [x\v]"                                               "";
    CASAB     "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"casab w1\v, w1\v, [x\v]"                                            "";
    CASAH     "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"casah w1\v, w1\v, [x\v]"                                            "";
    CASLB     "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"caslb w1\v, w1\v, [x\v]"                                            "";
    CASLH     "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"caslh w1\v, w1\v, [x\v]"                                            "";
    LDADDAB   "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldaddab w1\v, w1\v, [x\v]"                                          "";
    LDADDAH   "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldaddah w1\v, w1\v, [x\v]"                                          "";
    LDADDLB   "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldaddlb w1\v, w1\v, [x\v]"                                          "";
    LDADDLH   "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldaddlh w1\v, w1\v, [x\v]"                                          "";
    LDCLRAB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldclrab w1\v, w1\v, [x\v]"                                          "";
    LDCLRAH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldclrah w1\v, w1\v, [x\v]"                                          "";
    LDCLRLB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldclrlb w1\v, w1\v, [x\v]"                                          "";
    LDCLRLH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldclrlh w1\v, w1\v, [x\v]"                                          "";
    LDEORAB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldeorab w1\v, w1\v, [x\v]"                                          "";
    LDEORAH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldeorah w1\v, w1\v, [x\v]"                                          "";
    LDEORALB  "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldeoralb w1\v, w1\v, [x\v]"                                         "";
    LDEORALH  "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldeoralh w1\v, w1\v, [x\v]"                                         "";
    LDEORB    "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldeorb w1\v, w1\v, [x\v]"                                           "";
    LDEORH    "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldeorh w1\v, w1\v, [x\v]"                                           "";
    LDEORLB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldeorlb w1\v, w1\v, [x\v]"                                          "";
    LDEORLH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldeorlh w1\v, w1\v, [x\v]"                                          "";
    LDSETAB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldsetab w1\v, w1\v, [x\v]"                                          "";
    LDSETAH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldsetah w1\v, w1\v, [x\v]"                                          "";
    LDSETLB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldsetlb w1\v, w1\v, [x\v]"                                          "";
    LDSETLH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldsetlh w1\v, w1\v, [x\v]"                                          "";
    LDSMAXAB  "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsmaxab w1\v, w1\v, [x\v]"                                         "";
    LDSMAXAH  "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsmaxah w1\v, w1\v, [x\v]"                                         "";
    LDSMAXALB "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsmaxalb w1\v, w1\v, [x\v]"                                        "";
    LDSMAXALH "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsmaxalh w1\v, w1\v, [x\v]"                                        "";
    LDSMAXB   "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsmaxb w1\v, w1\v, [x\v]"                                          "";
    LDSMAXH   "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsmaxh w1\v, w1\v, [x\v]"                                          "";
    LDSMAXLB  "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsmaxlb w1\v, w1\v, [x\v]"                                         "";
    LDSMAXLH  "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsmaxlh w1\v, w1\v, [x\v]"                                         "";
    LDSMINAB  "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsminab w1\v, w1\v, [x\v]"                                         "";
    LDSMINAH  "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsminah w1\v, w1\v, [x\v]"                                         "";
    LDSMINALB "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsminalb w1\v, w1\v, [x\v]"                                        "";
    LDSMINALH "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsminalh w1\v, w1\v, [x\v]"                                        "";
    LDSMINB   "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsminb w1\v, w1\v, [x\v]"                                          "";
    LDSMINH   "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsminh w1\v, w1\v, [x\v]"                                          "";
    LDSMINLB  "w, w, [x]"                                  "i8"   "i8"   lse            atomics   lines           chained     r"ldsminlb w1\v, w1\v, [x\v]"                                         "";
    LDSMINLH  "w, w, [x]"                                  "i16"  "i16"  lse            atomics   lines           chained     r"ldsminlh w1\v, w1\v, [x\v]"                                         "";
    LDUMAXAB  "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldumaxab w1\v, w1\v, [x\v]"                                         "";
    LDUMAXAH  "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldumaxah w1\v, w1\v, [x\v]"                                         "";
    LDUMAXALB "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldumaxalb w1\v, w1\v, [x\v]"                                        "";
    LDUMAXALH "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldumaxalh w1\v, w1\v, [x\v]"                                        "";
    LDUMAXB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldumaxb w1\v, w1\v, [x\v]"                                          "";
    LDUMAXH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldumaxh w1\v, w1\v, [x\v]"                                          "";
    LDUMAXLB  "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"ldumaxlb w1\v, w1\v, [x\v]"                                         "";
    LDUMAXLH  "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"ldumaxlh w1\v, w1\v, [x\v]"                                         "";
    LDUMINAB  "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"lduminab w1\v, w1\v, [x\v]"                                         "";
    LDUMINAH  "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"lduminah w1\v, w1\v, [x\v]"                                         "";
    LDUMINALB "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"lduminalb w1\v, w1\v, [x\v]"                                        "";
    LDUMINALH "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"lduminalh w1\v, w1\v, [x\v]"                                        "";
    LDUMINB   "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"lduminb w1\v, w1\v, [x\v]"                                          "";
    LDUMINH   "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"lduminh w1\v, w1\v, [x\v]"                                          "";
    LDUMINLB  "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"lduminlb w1\v, w1\v, [x\v]"                                         "";
    LDUMINLH  "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"lduminlh w1\v, w1\v, [x\v]"                                         "";
    STADDB    "w, [x]"                                     "i8"   "i8"   lse            atomics   lines           throughput  r"staddb w1\v, [x\v]"                                                 "";
    STADDH    "w, [x]"                                     "i16"  "i16"  lse            atomics   lines           throughput  r"staddh w1\v, [x\v]"                                                 "";
    STADDLB   "w, [x]"                                     "i8"   "i8"   lse            atomics   lines           throughput  r"staddlb w1\v, [x\v]"                                                "";
    STADDLH   "w, [x]"                                     "i16"  "i16"  lse            atomics   lines           throughput  r"staddlh w1\v, [x\v]"                                                "";
    STCLRB    "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stclrb w1\v, [x\v]"                                                 "";
    STCLRH    "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stclrh w1\v, [x\v]"                                                 "";
    STCLRLB   "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stclrlb w1\v, [x\v]"                                                "";
    STCLRLH   "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stclrlh w1\v, [x\v]"                                                "";
    STEORB    "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"steorb w1\v, [x\v]"                                                 "";
    STEORH    "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"steorh w1\v, [x\v]"                                                 "";
    STEORLB   "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"steorlb w1\v, [x\v]"                                                "";
    STEORLH   "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"steorlh w1\v, [x\v]"                                                "";
    STSETB    "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stsetb w1\v, [x\v]"                                                 "";
    STSETH    "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stseth w1\v, [x\v]"                                                 "";
    STSETLB   "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stsetlb w1\v, [x\v]"                                                "";
    STSETLH   "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stsetlh w1\v, [x\v]"                                                "";
    STSMAXB   "w, [x]"                                     "i8"   "i8"   lse            atomics   lines           throughput  r"stsmaxb w1\v, [x\v]"                                                "";
    STSMAXH   "w, [x]"                                     "i16"  "i16"  lse            atomics   lines           throughput  r"stsmaxh w1\v, [x\v]"                                                "";
    STSMAXLB  "w, [x]"                                     "i8"   "i8"   lse            atomics   lines           throughput  r"stsmaxlb w1\v, [x\v]"                                               "";
    STSMAXLH  "w, [x]"                                     "i16"  "i16"  lse            atomics   lines           throughput  r"stsmaxlh w1\v, [x\v]"                                               "";
    STSMINB   "w, [x]"                                     "i8"   "i8"   lse            atomics   lines           throughput  r"stsminb w1\v, [x\v]"                                                "";
    STSMINH   "w, [x]"                                     "i16"  "i16"  lse            atomics   lines           throughput  r"stsminh w1\v, [x\v]"                                                "";
    STSMINLB  "w, [x]"                                     "i8"   "i8"   lse            atomics   lines           throughput  r"stsminlb w1\v, [x\v]"                                               "";
    STSMINLH  "w, [x]"                                     "i16"  "i16"  lse            atomics   lines           throughput  r"stsminlh w1\v, [x\v]"                                               "";
    STUMAXB   "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stumaxb w1\v, [x\v]"                                                "";
    STUMAXH   "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stumaxh w1\v, [x\v]"                                                "";
    STUMAXLB  "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stumaxlb w1\v, [x\v]"                                               "";
    STUMAXLH  "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stumaxlh w1\v, [x\v]"                                               "";
    STUMINB   "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stuminb w1\v, [x\v]"                                                "";
    STUMINH   "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stuminh w1\v, [x\v]"                                                "";
    STUMINLB  "w, [x]"                                     "u8"   "u8"   lse            atomics   lines           throughput  r"stuminlb w1\v, [x\v]"                                               "";
    STUMINLH  "w, [x]"                                     "u16"  "u16"  lse            atomics   lines           throughput  r"stuminlh w1\v, [x\v]"                                               "";
    SWPAB     "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"swpab w1\v, w1\v, [x\v]"                                            "";
    SWPAH     "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"swpah w1\v, w1\v, [x\v]"                                            "";
    SWPLB     "w, w, [x]"                                  "u8"   "u8"   lse            atomics   lines           chained     r"swplb w1\v, w1\v, [x\v]"                                            "";
    SWPLH     "w, w, [x]"                                  "u16"  "u16"  lse            atomics   lines           chained     r"swplh w1\v, w1\v, [x\v]"                                            "";

    LDAPR     "x, [x]"                                     "u64"  "u64"  rcpc           lrcpc     lines           chained     r"ldapr x\v, [x\v]"                                                   "";
    LDAPRB    "w, [x]"                                     "u8"   "u8"   rcpc           lrcpc     lines           throughput  r"ldaprb w1\v, [x\v]"                                                 "";
    LDAPRH    "w, [x]"                                     "u16"  "u16"  rcpc           lrcpc     lines           throughput  r"ldaprh w1\v, [x\v]"                                                 "";

    AND       "z.d, z.d, z.d"                              "u64"  "u64"  sve_streaming  sme       z               chained     r"and z\v\().d, z\v\().d, z\fixed\().d"                               "svand_n_u8_x svand_u64_x svand_u8_x svand_n_s32_x svand_n_u32_x svand_n_u16_x svand_n_u64_x";
    CMPEQ     "p.b, p/z, z.b, z.b"                         "i8"   "i8"   sve_streaming  sme       p               chained     r"cmpeq p\v\().b, p\v/z, z\fixed\().b, z\fixed\().b"                  "svcmpeq_n_u8 svcmpeq_u8";
    LD1B      "{z.b}, p/z, [x]"                            "u8"   "u8"   sve_streaming  sme       lines           throughput  r"ld1b {{z\v\().b}}, p0/z, [x\v]"                                     "svld1 svld1_s8 svld1_u8";
    AND       "p.b, p/z, p.b, p.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"and p\v\().b, p\fixed/z, p\v\().b, p\fixed\().b"                    "svand_b_z svunpkhi_b svunpklo_b";
    MOV       "z.s, s"                                     "u32"  "u32"  sve_streaming  sme       z               chained     r"mov z\v\().s, s\v"                                                  "svdup_f32 svdup_n_f32";
    CNTW      "x"                                          "u64"  "u64"  sve_streaming  sme       x               throughput  r"cntw x\v"                                                           "svcntw";
    WHILELO   "p.b, x, x"                                  "u64"  "u64"  sve_streaming  sme       p               throughput  r"whilelo p\v\().b, x8, x8"                                           "svwhilelt_b8 svwhilelt_b8_u64";
    PTRUE     "p.s"                                        "u32"  "u32"  sve_streaming  sme       p               throughput  r"ptrue p\v\().s"                                                     "svptrue_b32";
    ORR       "z.d, z.d, z.d"                              "u64"  "u64"  sve_streaming  sme       z               chained     r"orr z\v\().d, z\v\().d, z\fixed\().d"                               "svorr_u16_x svorr_u32_x svorr_u64_x svorr_u8_x";
    ORR       "p.b, p/z, p.b, p.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"orr p\v\().b, p\fixed/z, p\v\().b, p\fixed\().b"                    "svorr_b_z svorr_z";
    LD1W      "{z.s}, p/z, [x]"                            "u32"  "u32"  sve_streaming  sme       lines           throughput  r"ld1w {{z\v\().s}}, p0/z, [x\v]"                                     "svld1_f32 svld1_s32 svld1_u32";
    RDVL      "x, #imm"                                    "u64"  "u64"  sve_streaming  sme       x               throughput  r"rdvl x\v, #1"                                                       "svcntb";
    MOV       "z.s, w"                                     "u32"  "u32"  sve_streaming  sme       z               throughput  r"mov z\v\().s, w8"                                                   "svdup_n_u32 svdup_s32 svdup_u32";
    MOV       "z.d, d"                                     "u64"  "u64"  sve_streaming  sme       z               chained     r"mov z\v\().d, d\v"                                                  "svdup_f64 svdup_n_f64";
    WHILELO   "p.s, x, x"                                  "u64"  "u64"  sve_streaming  sme       p               throughput  r"whilelo p\v\().s, x8, x8"                                           "svwhilelt_b32_u64";
    PTRUE     "p.b"                                        "u8"   "u8"   sve_streaming  sme       p               throughput  r"ptrue p\v\().b"                                                     "svptrue_b8";
    SEL       "z.b, p, z.b, z.b"                           "u8"   "u8"   sve_streaming  sme       z               chained     r"sel z\v\().b, p0, z\v\().b, z\fixed\().b"                           "svsel_s8 svsel_u8 svand_u8_z svadd_u8_z svand_n_u8_z";
    CNTH      "x"                                          "u64"  "u64"  sve_streaming  sme       x               throughput  r"cnth x\v"                                                           "svcnth";
    MOV       "z.b, w"                                     "u8"   "u8"   sve_streaming  sme       z               throughput  r"mov z\v\().b, w8"                                                   "svdup_n_s8 svdup_n_u8 svdup_u8";
    WHILELO   "p.d, x, x"                                  "u64"  "u64"  sve_streaming  sme       p               throughput  r"whilelo p\v\().d, x8, x8"                                           "svwhilelt_b64 svwhilelt_b64_u64";
    ST1W      "{z.s}, p, [x]"                              "u32"  "u32"  sve_streaming  sme       lines           throughput  r"st1w {{z\v\().s}}, p0, [x\v]"                                       "svst1_f32 svst1_s32 svst1_u32";
    MOV       "z.b, p/m, w"                                "u8"   "u8"   sve_streaming  sme       z               chained     r"mov z\v\().b, p0/m, w8"                                             "svdup_n_u8_z svdup_u8_z";
    LD1H      "{z.h}, p/z, [x]"                            "u16"  "u16"  sve_streaming  sme       lines           throughput  r"ld1h {{z\v\().h}}, p0/z, [x\v]"                                     "svld1_bf16 svld1_f16 svld1_u16";
    PTEST     "p, p.b"                                     "u8"   "u8"   sve_streaming  sme       p               throughput  r"ptest p\fixed, p\v\().b"                                            "svptest_any svptest_first";
    LD1D      "{z.d}, p/z, [x]"                            "u64"  "u64"  sve_streaming  sme       lines           throughput  r"ld1d {{z\v\().d}}, p0/z, [x\v]"                                     "svld1_f64 svld1_s64 svld1_u64";
    EXT       "z.b, z.b, z.b, #imm"                        "u8"   "u8"   sve_streaming  sme       z               chained     r"ext z\v\().b, z\v\().b, z\fixed\().b, #1"                           "svext_f16 svext_f32 svext_u16 svext_u8";
    FSUB      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fsub z\v\().d, p0/m, z\v\().d, z\fixed\().d"                        "svsub_f64_x";
    CMPNE     "p.b, p/z, z.b, z.b"                         "i8"   "i8"   sve_streaming  sme       p               chained     r"cmpne p\v\().b, p\v/z, z\fixed\().b, z\fixed\().b"                  "svcmpne svcmpne_n_u8 svcmpne_u8";
    FADD      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fadd z\v\().d, p0/m, z\v\().d, z\fixed\().d"                        "svadd_f64_m svadd_f64_x";
    FCVT      "z.d, p/m, z.s"                              "f32"  "f64"  sve_streaming  sme       z               chained     r"fcvt z\v\().d, p0/m, z\v\().s"                                      "svcvt_f64_f32_x";
    PTRUE     "p.d"                                        "u64"  "u64"  sve_streaming  sme       p               throughput  r"ptrue p\v\().d"                                                     "svptrue_b64";
    CMPHI     "p.b, p/z, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"cmphi p\v\().b, p\v/z, z\fixed\().b, z\fixed\().b"                  "svcmpgt_n_u8 svcmplt_n_u8";
    BIC       "p.b, p/z, p.b, p.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"bic p\v\().b, p\fixed/z, p\v\().b, p\fixed\().b"                    "svbic_b_z";
    LSL       "z.b, p/m, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       z               chained     r"lsl z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "svlsl_n_s8_x svlsl_n_u8_x svlsl_u8_x";
    EOR       "z.d, z.d, z.d"                              "u64"  "u64"  sve_streaming  sme       z               chained     r"eor z\v\().d, z\v\().d, z\fixed\().d"                               "sveor_u16_x sveor_u32_x sveor_u64_x sveor_u8_x sveor_n_u8_x sveor_n_u64_x";
    ST1D      "{z.d}, p, [x]"                              "u64"  "u64"  sve_streaming  sme       lines           throughput  r"st1d {{z\v\().d}}, p0, [x\v]"                                       "svst1_f64 svst1_u64";
    LSR       "z.b, p/m, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       z               chained     r"lsr z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "svlsr_n_u8_x";
    CMPHS     "p.b, p/z, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"cmphs p\v\().b, p\v/z, z\fixed\().b, z\fixed\().b"                  "svcmpge_n_u8 svcmple_n_u8 svcmpge_u8 svcmple_u8";
    PTRUE     "p.h"                                        "u16"  "u16"  sve_streaming  sme       p               throughput  r"ptrue p\v\().h"                                                     "svptrue_b16";
    WHILELO   "p.h, x, x"                                  "u64"  "u64"  sve_streaming  sme       p               throughput  r"whilelo p\v\().h, x8, x8"                                           "svwhilelt_b16_u64";
    CNTD      "x"                                          "u64"  "u64"  sve_streaming  sme       x               throughput  r"cntd x\v"                                                           "svcntd";
    SEL       "z.s, p, z.s, z.s"                           "u32"  "u32"  sve_streaming  sme       z               chained     r"sel z\v\().s, p0, z\v\().s, z\fixed\().s"                           "svsel_f32 svsel_s32 svsel_u32";
    SUB       "z.b, z.b, z.b"                              "i8"   "i8"   sve_streaming  sme       z               chained     r"sub z\v\().b, z\v\().b, z\fixed\().b"                               "svsub_n_u8_x svsub_u8_x";
    FMLA      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmla z\v\().s, p0/m, z\fixed\().s, z\fixed\().s"                    "svmla_f32_m svmla_f32_x svmla_n_f32_x";
    ST1B      "{z.b}, p, [x]"                              "u8"   "u8"   sve_streaming  sme       lines           throughput  r"st1b {{z\v\().b}}, p0, [x\v]"                                       "svst1_s8 svst1_u8";
    FMUL      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmul z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svmul_f32_x svmul_n_f32_x";
    FSUB      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fsub z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svsub_f32_x";
    FCVT      "z.s, p/m, z.h"                              "f16"  "f32"  sve_streaming  sme       z               chained     r"fcvt z\v\().s, p0/m, z\v\().h"                                      "svcvt_f32_f16_x";
    TBL       "z.b, {z.b}, z.b"                            "u8"   "u8"   sve_streaming  sme       z               chained     r"tbl z\v\().b, {{z\v\().b}}, z\fixed\().b"                           "svtbl_s8 svtbl_u8";
    FMLA      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fmla z\v\().d, p0/m, z\fixed\().d, z\fixed\().d"                    "svmla_f64_m svmla_f64_x";
    MOV       "z.h, w"                                     "u16"  "u16"  sve_streaming  sme       z               throughput  r"mov z\v\().h, w8"                                                   "svdup_n_u16 svdup_u16";
    ADD       "z.b, p/m, z.b, z.b"                         "i8"   "i8"   sve_streaming  sme       z               chained     r"add z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "svadd_n_u8_m svadd_u8_z";
    ADD       "z.s, z.s, z.s"                              "i32"  "i32"  sve_streaming  sme       z               chained     r"add z\v\().s, z\v\().s, z\fixed\().s"                               "svadd_s32_x svadd_u32_x svadd_n_s32_x svadd_n_u32_x";
    FADD      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fadd z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svadd_f32_x svadd_n_f32_x";
    MOV       "z.d, x"                                     "u64"  "u64"  sve_streaming  sme       z               throughput  r"mov z\v\().d, x8"                                                   "svdup_n_s64 svdup_n_u64 svdup_s64 svdup_u64";
    LSL       "z.s, p/m, z.s, z.s"                         "u32"  "u32"  sve_streaming  sme       z               chained     r"lsl z\v\().s, p0/m, z\v\().s, z\fixed\().s"                         "svlsl_n_s32_x svlsl_n_u32_x svlsl_s32_x";
    LD1W      "{z.d}, p/z, [x]"                            "u32"  "u64"  sve_streaming  sme       lines           throughput  r"ld1w {{z\v\().d}}, p0/z, [x\v]"                                     "svld1uw_u64";
    PFALSE    "p.b"                                        "u8"   "u8"   sve_streaming  sme       p               throughput  r"pfalse p\v\().b"                                                    "svpfalse_b";
    UCVTF     "z.s, p/m, z.s"                              "u32"  "f32"  sve_streaming  sme       z               chained     r"ucvtf z\v\().s, p0/m, z\v\().s"                                     "svcvt_f32_u32_x";
    FMAX      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmax z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svmax_f32_x svmax_n_f32_x";
    CNTP      "x, p, p.b"                                  "u8"   "u8"   sve_streaming  sme       p               throughput  r"cntp x8, p\fixed, p\v\().b"                                         "svcntp_b8";
    SUB       "z.s, z.s, z.s"                              "i32"  "i32"  sve_streaming  sme       z               chained     r"sub z\v\().s, z\v\().s, z\fixed\().s"                               "svsub_s32_x svsub_u32_x svsubr_n_s32_x";
    FMUL      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fmul z\v\().d, p0/m, z\v\().d, z\fixed\().d"                        "svmul_f64_x";
    UUNPKLO   "z.h, z.b"                                   "u8"   "u16"  sve_streaming  sme       z               chained     r"uunpklo z\v\().h, z\v\().b"                                         "svunpklo_u16";
    FMAD      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmad z\v\().s, p0/m, z\fixed\().s, z\fixed\().s"                    "svmad_f32_x svmad_n_f32_x";
    CNTP      "x, p, p.s"                                  "u32"  "u32"  sve_streaming  sme       p               throughput  r"cntp x8, p\fixed, p\v\().s"                                         "svcntp_b32";
    LD1RQB    "{z.b}, p/z, [x]"                            "u8"   "u8"   sve_streaming  sme       lines           throughput  r"ld1rqb {{z\v\().b}}, p0/z, [x\v]"                                   "svld1rq_u8";
    ST1H      "{z.h}, p, [x]"                              "u16"  "u16"  sve_streaming  sme       lines           throughput  r"st1h {{z\v\().h}}, p0, [x\v]"                                       "svst1_f16 svst1_u16";
    LSR       "z.s, p/m, z.s, z.s"                         "u32"  "u32"  sve_streaming  sme       z               chained     r"lsr z\v\().s, p0/m, z\v\().s, z\fixed\().s"                         "svlsr_n_u32_x";
    FDIV      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fdiv z\v\().s, p0/m, z\v\().s, z\v\().s"                            "svdiv_f32_x svdiv_n_f32_x";
    BRKB      "p.b, p/z, p.b"                              "u8"   "u8"   sve_streaming  sme       p               chained     r"brkb p\v\().b, p\fixed/z, p\v\().b"                                 "svbrkb_b_z";
    LASTB     "w, p, z.s"                                  "u32"  "u32"  sve_streaming  sme       z               throughput  r"lastb w8, p0, z\v\().s"                                             "svlastb svlastb_u32";
    UUNPKHI   "z.h, z.b"                                   "u8"   "u16"  sve_streaming  sme       z               chained     r"uunpkhi z\v\().h, z\v\().b"                                         "svunpkhi_u16";
    INDEX     "z.b, w, w"                                  "i8"   "i8"   sve_streaming  sme       z               throughput  r"index z\v\().b, w8, w8"                                             "svindex_u8";
    ADD       "z.b, z.b, z.b"                              "i8"   "i8"   sve_streaming  sme       z               chained     r"add z\v\().b, z\v\().b, z\fixed\().b"                               "svadd_u8_x svadd_n_u8_x";
    UUNPKHI   "z.s, z.h"                                   "u16"  "u32"  sve_streaming  sme       z               chained     r"uunpkhi z\v\().s, z\v\().h"                                         "svunpkhi_u32";
    UUNPKLO   "z.s, z.h"                                   "u16"  "u32"  sve_streaming  sme       z               chained     r"uunpklo z\v\().s, z\v\().h"                                         "svunpklo_u32";
    ADD       "z.d, z.d, z.d"                              "i64"  "i64"  sve_streaming  sme       z               chained     r"add z\v\().d, z\v\().d, z\fixed\().d"                               "svadd_n_u64_x svadd_s64_x svadd_u64_x";
    INDEX     "z.d, x, x"                                  "i64"  "i64"  sve_streaming  sme       z               throughput  r"index z\v\().d, x8, x8"                                             "svindex_u64";
    SCVTF     "z.s, p/m, z.s"                              "i32"  "f32"  sve_streaming  sme       z               chained     r"scvtf z\v\().s, p0/m, z\v\().s"                                     "svcvt_f32_s32_x";
    UDOT      "z.s, z.b, z.b"                              "u8"   "u32"  sve_streaming  sme       z               chained     r"udot z\v\().s, z\fixed\().b, z\fixed\().b"                          "svdot_u32";
    CNTP      "x, p, p.d"                                  "u64"  "u64"  sve_streaming  sme       p               throughput  r"cntp x8, p\fixed, p\v\().d"                                         "svcntp_b64";
    FNMLS     "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fnmls z\v\().d, p0/m, z\fixed\().d, z\fixed\().d"                   "svnmls_f64_x";
    LASTB     "w, p, z.b"                                  "u8"   "u8"   sve_streaming  sme       z               throughput  r"lastb w8, p0, z\v\().b"                                             "svlastb_u8";
    MOV       "z.h, h"                                     "u16"  "u16"  sve_streaming  sme       z               chained     r"mov z\v\().h, h\v"                                                  "svdup_f16 svdup_n_f16";
    CMPNE     "p.s, p/z, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       p               chained     r"cmpne p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpne_n_u32";
    REV       "z.s, z.s"                                   "u32"  "u32"  sve_streaming  sme       z               chained     r"rev z\v\().s, z\v\().s"                                             "svrev_u32";
    SEL       "z.d, p, z.d, z.d"                           "u64"  "u64"  sve_streaming  sme       z               chained     r"sel z\v\().d, p0, z\v\().d, z\fixed\().d"                           "svsel_f64 svsel_s64 svsel_u64";
    TBL       "z.d, {z.d}, z.d"                            "u64"  "u64"  sve_streaming  sme       z               chained     r"tbl z\v\().d, {{z\v\().d}}, z\fixed\().d"                           "svtbl_f64 svtbl_u64 svdup_lane_s64";
    FSCALE    "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fscale z\v\().s, p0/m, z\v\().s, z\fixed\().s"                      "svscale_f32_x";
    FSQRT     "z.s, p/m, z.s"                              "f32"  "f32"  sve_streaming  sme       z               chained     r"fsqrt z\v\().s, p0/m, z\v\().s"                                     "svsqrt_f32_x";
    PTRUE     "p.b, vl1"                                   "u8"   "u8"   sve_streaming  sme       p               throughput  r"ptrue p\v\().b, vl1"                                                "svptrue_pat_b8";
    REV       "p.b, p.b"                                   "u8"   "u8"   sve_streaming  sme       p               chained     r"rev p\v\().b, p\v\().b"                                             "svrev_b8";
    SDOT      "z.s, z.b, z.b"                              "i8"   "i32"  sve_streaming  sme       z               chained     r"sdot z\v\().s, z\fixed\().b, z\fixed\().b"                          "svdot_s32";
    ZIP1      "z.h, z.h, z.h"                              "u16"  "u16"  sve_streaming  sme       z               chained     r"zip1 z\v\().h, z\v\().h, z\fixed\().h"                              "svzip1_u16";
    AND       "z.b, p/m, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       z               chained     r"and z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "svand_u8_z svand_n_u8_z svand_u8_m";
    ORR       "z.b, p/m, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       z               chained     r"orr z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "svorr_u8_m svorr_n_u8_m";
    SMAX      "z.s, p/m, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       z               chained     r"smax z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svmax_s32_x svmax_n_s32_x";
    LASTA     "w, p, z.s"                                  "u32"  "u32"  sve_streaming  sme       z               throughput  r"lasta w8, p0, z\v\().s"                                             "svlasta";
    ZIP1      "z.b, z.b, z.b"                              "u8"   "u8"   sve_streaming  sme       z               chained     r"zip1 z\v\().b, z\v\().b, z\fixed\().b"                              "svzip1_u8";
    CMPHI     "p.d, p/z, z.d, z.d"                         "u64"  "u64"  sve_streaming  sme       p               chained     r"cmphi p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmplt_n_u64 svcmpgt_u64 svcmplt_u64";
    FCMNE     "p.s, p/z, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       p               chained     r"fcmne p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpne_f32";
    FCVTZS    "z.s, p/m, z.s"                              "f32"  "i32"  sve_streaming  sme       z               chained     r"fcvtzs z\v\().s, p0/m, z\v\().s"                                    "svcvt_s32_f32_x";
    INDEX     "z.s, w, w"                                  "i32"  "i32"  sve_streaming  sme       z               throughput  r"index z\v\().s, w8, w8"                                             "svindex_u32";
    CMPHS     "p.s, p/z, z.s, z.s"                         "u32"  "u32"  sve_streaming  sme       p               chained     r"cmphs p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpge_n_u32 svcmple_n_u32";
    UZP1      "z.h, z.h, z.h"                              "u16"  "u16"  sve_streaming  sme       z               chained     r"uzp1 z\v\().h, z\v\().h, z\fixed\().h"                              "svuzp1_s16 svuzp1_u16";
    CMPEQ     "p.s, p/z, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       p               chained     r"cmpeq p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpeq_n_u32 svcmpeq_u32";
    SUB       "z.d, z.d, z.d"                              "i64"  "i64"  sve_streaming  sme       z               chained     r"sub z\v\().d, z\v\().d, z\fixed\().d"                               "svsub_u64_x svsub_n_s64_x";
    FMUL      "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fmul z\v\().h, p0/m, z\v\().h, z\fixed\().h"                        "svmul_f16_x svmul_n_f16_x";
    FRINTN    "z.s, p/m, z.s"                              "f32"  "f32"  sve_streaming  sme       z               chained     r"frintn z\v\().s, p0/m, z\v\().s"                                    "svrintn_f32_x";
    LASTB     "x, p, z.d"                                  "u64"  "u64"  sve_streaming  sme       z               throughput  r"lastb x8, p0, z\v\().d"                                             "svlastb svlastb_u64";
    LD1B      "{z.h}, p/z, [x]"                            "u8"   "u16"  sve_streaming  sme       lines           throughput  r"ld1b {{z\v\().h}}, p0/z, [x\v]"                                     "svld1ub_u16";
    LSR       "z.d, p/m, z.d, z.d"                         "u64"  "u64"  sve_streaming  sme       z               chained     r"lsr z\v\().d, p0/m, z\v\().d, z\fixed\().d"                         "svlsr_n_u64_x";
    LSL       "z.h, p/m, z.h, z.h"                         "u16"  "u16"  sve_streaming  sme       z               chained     r"lsl z\v\().h, p0/m, z\v\().h, z\fixed\().h"                         "svlsl_n_u16_x";
    SPLICE    "z.h, p, z.h, z.h"                           "u16"  "u16"  sve_streaming  sme       z               chained     r"splice z\v\().h, p0, z\v\().h, z\fixed\().h"                        "svsplice_u16";
    FMLS      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmls z\v\().s, p0/m, z\fixed\().s, z\fixed\().s"                    "svmls_f32_m svmls_f32_x svmls_n_f32_x";
    LSL       "z.d, p/m, z.d, z.d"                         "u64"  "u64"  sve_streaming  sme       z               chained     r"lsl z\v\().d, p0/m, z\v\().d, z\fixed\().d"                         "svlsl_u64_x svlsl_n_s64_x svlsl_n_u64_x";
    MUL       "z.s, p/m, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       z               chained     r"mul z\v\().s, p0/m, z\v\().s, z\fixed\().s"                         "svmul_n_s32_x svmul_n_u32_x svmul_s32_x";
    FMLA      "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fmla z\v\().h, p0/m, z\fixed\().h, z\fixed\().h"                    "svmla_f16_x";
    FSCALE    "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fscale z\v\().d, p0/m, z\v\().d, z\fixed\().d"                      "svscale_f64_x";
    LD1H      "{z.s}, p/z, [x]"                            "u16"  "u32"  sve_streaming  sme       lines           throughput  r"ld1h {{z\v\().s}}, p0/z, [x\v]"                                     "svld1uh_u32";
    FCMEQ     "p.s, p/z, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       p               chained     r"fcmeq p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpeq_n_f32";
    FCMGT     "p.s, p/z, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       p               chained     r"fcmgt p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpgt_n_f32";
    NOT       "p.b, p/z, p.b"                              "u8"   "u8"   sve_streaming  sme       p               chained     r"not p\v\().b, p\fixed/z, p\v\().b"                                  "svnot_b_z";
    ORV       "d, p, z.d"                                  "u64"  "u64"  sve_streaming  sme       z               chained     r"orv d\v, p0, z\v\().d"                                              "svorv_u64";
    SEL       "z.h, p, z.h, z.h"                           "u16"  "u16"  sve_streaming  sme       z               chained     r"sel z\v\().h, p0, z\v\().h, z\fixed\().h"                           "svsel_u16";
    FMLS      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fmls z\v\().d, p0/m, z\fixed\().d, z\fixed\().d"                    "svmls_f64_m svmls_n_f64_x";
    BFDOT     "z.s, z.h, z.h"                              "bf16" "f32"  sve_streaming  sme       z               chained     r"bfdot z\v\().s, z\fixed\().h, z\fixed\().h"                         "svbfdot_f32";
    FCVT      "z.h, p/m, z.s"                              "f32"  "f16"  sve_streaming  sme       z               chained     r"fcvt z\v\().h, p0/m, z\v\().s"                                      "svcvt_f16_f32_x";
    FSQRT     "z.d, p/m, z.d"                              "f64"  "f64"  sve_streaming  sme       z               chained     r"fsqrt z\v\().d, p0/m, z\v\().d"                                     "svsqrt_f64_x";
    LASTA     "x, p, z.d"                                  "u64"  "u64"  sve_streaming  sme       z               throughput  r"lasta x8, p0, z\v\().d"                                             "svlasta svlasta_u64";
    LD2D      "{z.d, z.d}, p/z, [x]"                       "u64"  "u64"  sve_streaming  sme       lines           throughput  r"ld2d {{z16.d, z17.d}}, p0/z, [x\v]"                                 "svld2_f64 svld2_u64";
    FCMEQ     "p.d, p/z, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       p               chained     r"fcmeq p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmpeq_n_f64";
    ASR       "z.s, p/m, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       z               chained     r"asr z\v\().s, p0/m, z\v\().s, z\fixed\().s"                         "svasr_n_s32_x svasr_s32_x";
    CNTP      "x, p, p.h"                                  "u16"  "u16"  sve_streaming  sme       p               throughput  r"cntp x8, p\fixed, p\v\().h"                                         "svcntp_b16";
    FDIV      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fdiv z\v\().d, p0/m, z\v\().d, z\v\().d"                            "svdiv_f64_x";
    FMAXNM    "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmaxnm z\v\().s, p0/m, z\v\().s, z\fixed\().s"                      "svmaxnm_f32_x";
    FNEG      "z.d, p/m, z.d"                              "f64"  "f64"  sve_streaming  sme       z               chained     r"fneg z\v\().d, p0/m, z\v\().d"                                      "svneg_f64_x";
    FNEG      "z.s, p/m, z.s"                              "f32"  "f32"  sve_streaming  sme       z               chained     r"fneg z\v\().s, p0/m, z\v\().s"                                      "svneg_f32_x";
    LASTB     "d, p, z.d"                                  "u64"  "u64"  sve_streaming  sme       z               chained     r"lastb d\v, p0, z\v\().d"                                            "svlastb_f64";
    LASTB     "w, p, z.h"                                  "u16"  "u16"  sve_streaming  sme       z               throughput  r"lastb w8, p0, z\v\().h"                                             "svlastb";
    LD2H      "{z.h, z.h}, p/z, [x]"                       "u16"  "u16"  sve_streaming  sme       lines           throughput  r"ld2h {{z16.h, z17.h}}, p0/z, [x\v]"                                 "svld2_f16";
    LD2W      "{z.s, z.s}, p/z, [x]"                       "u32"  "u32"  sve_streaming  sme       lines           throughput  r"ld2w {{z16.s, z17.s}}, p0/z, [x\v]"                                 "svld2_f32";
    ASR       "z.b, p/m, z.b, z.b"                         "i8"   "i8"   sve_streaming  sme       z               chained     r"asr z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "svasr_n_s8_x";
    CMPEQ     "p.d, p/z, z.d, z.d"                         "i64"  "i64"  sve_streaming  sme       p               chained     r"cmpeq p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmpeq_n_u64";
    NEG       "z.d, p/m, z.d"                              "i64"  "i64"  sve_streaming  sme       z               chained     r"neg z\v\().d, p0/m, z\v\().d"                                       "svneg_s64_x";
    SCVTF     "z.d, p/m, z.d"                              "i64"  "f64"  sve_streaming  sme       z               chained     r"scvtf z\v\().d, p0/m, z\v\().d"                                     "svcvt_f64_s64_x";
    TRN1      "z.s, z.s, z.s"                              "u32"  "u32"  sve_streaming  sme       z               chained     r"trn1 z\v\().s, z\v\().s, z\fixed\().s"                              "svtrn1_f32";
    UZP1      "z.b, z.b, z.b"                              "u8"   "u8"   sve_streaming  sme       z               chained     r"uzp1 z\v\().b, z\v\().b, z\fixed\().b"                              "svuzp1_s8 svuzp1_u8";
    CNT       "z.b, p/m, z.b"                              "u8"   "u8"   sve_streaming  sme       z               chained     r"cnt z\v\().b, p0/m, z\v\().b"                                       "svcnt_u8_x svcnt_u8_z";
    FMAD      "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fmad z\v\().h, p0/m, z\fixed\().h, z\fixed\().h"                    "svmad_f16_x svmad_n_f16_x";
    EOR       "p.b, p/z, p.b, p.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"eor p\v\().b, p\fixed/z, p\v\().b, p\fixed\().b"                    "sveor_b_z";
    INSR      "z.d, x"                                     "u64"  "u64"  sve_streaming  sme       z               chained     r"insr z\v\().d, x8"                                                  "svinsr_n_s64 svinsr_n_u64";
    FACGT     "p.h, p/z, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       p               chained     r"facgt p\v\().h, p\v/z, z\fixed\().h, z\fixed\().h"                  "svacgt_n_f16";
    FMLS      "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fmls z\v\().h, p0/m, z\fixed\().h, z\fixed\().h"                    "svmls_n_f16_x";
    FSUBR     "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fsubr z\v\().s, p0/m, z\v\().s, z\fixed\().s"                       "svsubr_n_f32_x";
    REV       "z.d, z.d"                                   "u64"  "u64"  sve_streaming  sme       z               chained     r"rev z\v\().d, z\v\().d"                                             "svrev_u64";
    UUNPKHI   "z.d, z.s"                                   "u32"  "u64"  sve_streaming  sme       z               chained     r"uunpkhi z\v\().d, z\v\().s"                                         "svunpkhi_u64";
    UUNPKLO   "z.d, z.s"                                   "u32"  "u64"  sve_streaming  sme       z               chained     r"uunpklo z\v\().d, z\v\().s"                                         "svunpklo_u64";
    EOR       "z.b, p/m, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       z               chained     r"eor z\v\().b, p0/m, z\v\().b, z\fixed\().b"                         "sveor_n_u8_m sveor_u8_m";
    ADD       "z.h, z.h, z.h"                              "i16"  "i16"  sve_streaming  sme       z               chained     r"add z\v\().h, z\v\().h, z\fixed\().h"                               "svadd_n_u16_x svadd_u16_x";
    BRKA      "p.b, p/z, p.b"                              "u8"   "u8"   sve_streaming  sme       p               chained     r"brka p\v\().b, p\fixed/z, p\v\().b"                                 "svbrka_b_z";
    FCMUO     "p.d, p/z, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       p               chained     r"fcmuo p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmpuo_f64";
    FCMUO     "p.s, p/z, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       p               chained     r"fcmuo p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpuo_f32";
    FCVT      "z.s, p/m, z.d"                              "f64"  "f32"  sve_streaming  sme       z               chained     r"fcvt z\v\().s, p0/m, z\v\().d"                                      "svcvt_f32_f64_x";
    FCVTZS    "z.d, p/m, z.d"                              "f64"  "i64"  sve_streaming  sme       z               chained     r"fcvtzs z\v\().d, p0/m, z\v\().d"                                    "svcvt_s64_f64_x";
    FCVTZS    "z.h, p/m, z.h"                              "f16"  "i16"  sve_streaming  sme       z               chained     r"fcvtzs z\v\().h, p0/m, z\v\().h"                                    "svcvt_s16_f16_x";
    FMAX      "z.d, p/m, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       z               chained     r"fmax z\v\().d, p0/m, z\v\().d, z\fixed\().d"                        "svmax_f64_x";
    FRINTN    "z.d, p/m, z.d"                              "f64"  "f64"  sve_streaming  sme       z               chained     r"frintn z\v\().d, p0/m, z\v\().d"                                    "svrintn_f64_x";
    FRINTN    "z.h, p/m, z.h"                              "f16"  "f16"  sve_streaming  sme       z               chained     r"frintn z\v\().h, p0/m, z\v\().h"                                    "svrintn_f16_x";
    INSR      "z.b, w"                                     "u8"   "u8"   sve_streaming  sme       z               chained     r"insr z\v\().b, w8"                                                  "svinsr_n_u8";
    LASTA     "w, p, z.h"                                  "u16"  "u16"  sve_streaming  sme       z               throughput  r"lasta w8, p0, z\v\().h"                                             "svlasta";
    CMPGT     "p.b, p/z, z.b, z.b"                         "i8"   "i8"   sve_streaming  sme       p               chained     r"cmpgt p\v\().b, p\v/z, z\fixed\().b, z\fixed\().b"                  "svcmplt_n_s8";
    ASR       "z.d, p/m, z.d, z.d"                         "i64"  "i64"  sve_streaming  sme       z               chained     r"asr z\v\().d, p0/m, z\v\().d, z\fixed\().d"                         "svasr_n_s64_x";
    CMPHS     "p.d, p/z, z.d, z.d"                         "u64"  "u64"  sve_streaming  sme       p               chained     r"cmphs p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmple_n_u64";
    MUL       "z.d, p/m, z.d, z.d"                         "i64"  "i64"  sve_streaming  sme       z               chained     r"mul z\v\().d, p0/m, z\v\().d, z\fixed\().d"                         "svmul_n_u64_x";
    FADD      "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fadd z\v\().h, p0/m, z\v\().h, z\fixed\().h"                        "svadd_n_f16_m svadd_n_f16_x";
    ADD       "z.h, p/m, z.h, z.h"                         "i16"  "i16"  sve_streaming  sme       z               chained     r"add z\v\().h, p0/m, z\v\().h, z\fixed\().h"                         "svadd_n_u16_m";
    CMPHS     "p.h, p/z, z.h, z.h"                         "u16"  "u16"  sve_streaming  sme       p               chained     r"cmphs p\v\().h, p\v/z, z\fixed\().h, z\fixed\().h"                  "svcmple_n_u16";
    CMPNE     "p.h, p/z, z.h, z.h"                         "i16"  "i16"  sve_streaming  sme       p               chained     r"cmpne p\v\().h, p\v/z, z\fixed\().h, z\fixed\().h"                  "svcmpne_n_u16";
    FACGE     "p.s, p/z, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       p               chained     r"facge p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svacge_n_f32";
    CMPGT     "p.s, p/z, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       p               chained     r"cmpgt p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpgt_n_s32 svcmplt_n_s32";
    CMPHI     "p.s, p/z, z.s, z.s"                         "u32"  "u32"  sve_streaming  sme       p               chained     r"cmphi p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpgt_n_u32 svcmplt_n_u32";
    PNEXT     "p.b, p, p.b"                                "u8"   "u8"   sve_streaming  sme       p               chained     r"pnext p\v\().b, p\fixed, p\v\().b"                                  "svpnext_b8";
    REV       "z.b, z.b"                                   "u8"   "u8"   sve_streaming  sme       z               chained     r"rev z\v\().b, z\v\().b"                                             "svrev_u8";
    REVB      "z.s, p/m, z.s"                              "u32"  "u32"  sve_streaming  sme       z               chained     r"revb z\v\().s, p0/m, z\v\().s"                                      "svrevb_u32_x";
    SCVTF     "z.s, p/m, z.d"                              "i64"  "f32"  sve_streaming  sme       z               chained     r"scvtf z\v\().s, p0/m, z\v\().d"                                     "svcvt_f32_s64_x";
    ST1H      "{z.s}, p, [x]"                              "u16"  "u32"  sve_streaming  sme       lines           throughput  r"st1h {{z\v\().s}}, p0, [x\v]"                                       "svst1h_u32";
    ST1W      "{z.d}, p, [x]"                              "u32"  "u64"  sve_streaming  sme       lines           throughput  r"st1w {{z\v\().d}}, p0, [x\v]"                                       "svst1w_u64";
    TRN1      "z.h, z.h, z.h"                              "u16"  "u16"  sve_streaming  sme       z               chained     r"trn1 z\v\().h, z\v\().h, z\fixed\().h"                              "svtrn1_f16";
    TRN2      "z.s, z.s, z.s"                              "u32"  "u32"  sve_streaming  sme       z               chained     r"trn2 z\v\().s, z\v\().s, z\fixed\().s"                              "svtrn2_f32";
    UADDV     "d, p, z.b"                                  "u8"   "u64"  sve_streaming  sme       z               chained     r"uaddv d\v, p0, z\v\().b"                                            "svaddv_u8";
    UADDV     "d, p, z.d"                                  "u64"  "u64"  sve_streaming  sme       z               chained     r"uaddv d\v, p0, z\v\().d"                                            "svaddv_u64";
    UADDV     "d, p, z.h"                                  "u16"  "u64"  sve_streaming  sme       z               chained     r"uaddv d\v, p0, z\v\().h"                                            "svaddv_u16";
    UADDV     "d, p, z.s"                                  "u32"  "u64"  sve_streaming  sme       z               chained     r"uaddv d\v, p0, z\v\().s"                                            "svaddv_u32";
    UCVTF     "z.s, p/m, z.d"                              "u64"  "f32"  sve_streaming  sme       z               chained     r"ucvtf z\v\().s, p0/m, z\v\().d"                                     "svcvt_f32_u64_x";
    UMAX      "z.d, p/m, z.d, z.d"                         "u64"  "u64"  sve_streaming  sme       z               chained     r"umax z\v\().d, p0/m, z\v\().d, z\fixed\().d"                        "svmax_u64_m svmax_u64_x";
    UQINCP    "x, p.d"                                     "u64"  "u64"  sve_streaming  sme       x               chained     r"uqincp x\v, p0.d"                                                   "svqincp_n_u64_b64";
    CMPEQ     "p.h, p/z, z.h, z.h"                         "i16"  "i16"  sve_streaming  sme       p               chained     r"cmpeq p\v\().h, p\v/z, z\fixed\().h, z\fixed\().h"                  "svcmpeq_u16 svcmpeq_n_u16";
    FCMGE     "p.s, p/z, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       p               chained     r"fcmge p\v\().s, p\v/z, z\fixed\().s, z\fixed\().s"                  "svcmpge_f32 svcmpge_n_f32";
    UQSUB     "z.h, z.h, z.h"                              "u16"  "u16"  sve_streaming  sme       z               chained     r"uqsub z\v\().h, z\v\().h, z\fixed\().h"                             "svqsub_n_u16 svqsub_u16";
    CNT       "z.s, p/m, z.s"                              "u32"  "u32"  sve_streaming  sme       z               chained     r"cnt z\v\().s, p0/m, z\v\().s"                                       "svcnt_u32_x";
    CNTB      "x"                                          "u64"  "u64"  sve_streaming  sme       x               throughput  r"cntb x\v"                                                           "";
    FABS      "z.d, p/m, z.d"                              "f64"  "f64"  sve_streaming  sme       z               chained     r"fabs z\v\().d, p0/m, z\v\().d"                                      "svabs_f64_x";
    FABS      "z.h, p/m, z.h"                              "f16"  "f16"  sve_streaming  sme       z               chained     r"fabs z\v\().h, p0/m, z\v\().h"                                      "svabs_f16_x";
    FABS      "z.s, p/m, z.s"                              "f32"  "f32"  sve_streaming  sme       z               chained     r"fabs z\v\().s, p0/m, z\v\().s"                                      "svabs_f32_x";
    FADDV     "d, p, z.d"                                  "f64"  "f64"  sve_streaming  sme       z               chained     r"faddv d\v, p0, z\v\().d"                                            "svaddv_f64";
    FADDV     "s, p, z.s"                                  "f32"  "f32"  sve_streaming  sme       z               chained     r"faddv s\v, p0, z\v\().s"                                            "svaddv_f32";
    FCMGE     "p.d, p/z, z.d, z.d"                         "f64"  "f64"  sve_streaming  sme       p               chained     r"fcmge p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmpge_f64";
    FMIN      "z.s, p/m, z.s, z.s"                         "f32"  "f32"  sve_streaming  sme       z               chained     r"fmin z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svmin_f32_x";
    FNEG      "z.h, p/m, z.h"                              "f16"  "f16"  sve_streaming  sme       z               chained     r"fneg z\v\().h, p0/m, z\v\().h"                                      "svneg_f16_m";
    LASTA     "w, p, z.b"                                  "u8"   "u8"   sve_streaming  sme       z               throughput  r"lasta w8, p0, z\v\().b"                                             "svlasta_u8";
    LD3B      "{z.b - z.b}, p/z, [x]"                      "u8"   "u8"   sve_streaming  sme       lines           throughput  r"ld3b {{z16.b-z18.b}}, p0/z, [x\v]"                                  "svld3_u8";
    MOV       "p.b, p/z, p.b"                              "u8"   "u8"   sve_streaming  sme       p               chained     r"mov p\v\().b, p\fixed/z, p\v\().b"                                  "svmov_b_z";
    CMPGT     "p.d, p/z, z.d, z.d"                         "i64"  "i64"  sve_streaming  sme       p               chained     r"cmpgt p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmplt_n_s64";
    CMPNE     "p.d, p/z, z.d, z.d"                         "i64"  "i64"  sve_streaming  sme       p               chained     r"cmpne p\v\().d, p\v/z, z\fixed\().d, z\fixed\().d"                  "svcmpne_n_u64";
    SMAX      "z.d, p/m, z.d, z.d"                         "i64"  "i64"  sve_streaming  sme       z               chained     r"smax z\v\().d, p0/m, z\v\().d, z\fixed\().d"                        "svmax_n_s64_x";
    FDIVR     "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fdivr z\v\().h, p0/m, z\v\().h, z\fixed\().h"                       "svdivr_n_f16_m";
    FSUBR     "z.h, p/m, z.h, z.h"                         "f16"  "f16"  sve_streaming  sme       z               chained     r"fsubr z\v\().h, p0/m, z\v\().h, z\fixed\().h"                       "svsubr_n_f16_m";
    MOV       "z.h, p/m, w"                                "u16"  "u16"  sve_streaming  sme       z               chained     r"mov z\v\().h, p0/m, w8"                                             "svdup_n_u16_m";
    CMPHI     "p.h, p/z, z.h, z.h"                         "u16"  "u16"  sve_streaming  sme       p               chained     r"cmphi p\v\().h, p\v/z, z\fixed\().h, z\fixed\().h"                  "svcmpgt_n_u16";
    SUB       "z.h, p/m, z.h, z.h"                         "i16"  "i16"  sve_streaming  sme       z               chained     r"sub z\v\().h, p0/m, z\v\().h, z\fixed\().h"                         "svsub_n_u16_m";
    SUB       "z.h, z.h, z.h"                              "i16"  "i16"  sve_streaming  sme       z               chained     r"sub z\v\().h, z\v\().h, z\fixed\().h"                               "svsubr_n_u16_x";
    SMIN      "z.s, p/m, z.s, z.s"                         "i32"  "i32"  sve_streaming  sme       z               chained     r"smin z\v\().s, p0/m, z\v\().s, z\fixed\().s"                        "svmin_n_s32_x";
    TBL       "z.s, {z.s}, z.s"                            "u32"  "u32"  sve_streaming  sme       z               chained     r"tbl z\v\().s, {{z\v\().s}}, z\fixed\().s"                           "svdup_lane_f32";
    MOV       "z.s, p/m, w"                                "u32"  "u32"  sve_streaming  sme       z               chained     r"mov z\v\().s, p0/m, w8"                                             "svdup_n_u32_z";
    NEG       "z.b, p/m, z.b"                              "i8"   "i8"   sve_streaming  sme       z               chained     r"neg z\v\().b, p0/m, z\v\().b"                                       "svneg_s8_x";
    NOR       "p.b, p/z, p.b, p.b"                         "u8"   "u8"   sve_streaming  sme       p               chained     r"nor p\v\().b, p\fixed/z, p\v\().b, p\fixed\().b"                    "svnor_b_z";
    NOT       "z.h, p/m, z.h"                              "u16"  "u16"  sve_streaming  sme       z               chained     r"not z\v\().h, p0/m, z\v\().h"                                       "svnot_s16_x";
    NOT       "z.s, p/m, z.s"                              "u32"  "u32"  sve_streaming  sme       z               chained     r"not z\v\().s, p0/m, z\v\().s"                                       "svnot_s32_x";
    ORV       "b, p, z.b"                                  "u8"   "u8"   sve_streaming  sme       z               chained     r"orv b\v, p0, z\v\().b"                                              "svorv_u8";
    PUNPKHI   "p.h, p.b"                                   "u8"   "u8"   sve_streaming  sme       p               chained     r"punpkhi p\v\().h, p\v\().b"                                         "svunpkhi_b";
    PUNPKLO   "p.h, p.b"                                   "u8"   "u8"   sve_streaming  sme       p               chained     r"punpklo p\v\().h, p\v\().b"                                         "svunpklo_b";
    RBIT      "z.b, p/m, z.b"                              "u8"   "u8"   sve_streaming  sme       z               chained     r"rbit z\v\().b, p0/m, z\v\().b"                                      "svrbit_u8_x";
    SABD      "z.b, p/m, z.b, z.b"                         "i8"   "i8"   sve_streaming  sme       z               chained     r"sabd z\v\().b, p0/m, z\v\().b, z\fixed\().b"                        "svabd_s8_x";
    SADDV     "d, p, z.s"                                  "i32"  "i64"  sve_streaming  sme       z               chained     r"saddv d\v, p0, z\v\().s"                                            "svaddv_s32";
    SEL       "p.b, p, p.b, p.b"                           "u8"   "u8"   sve_streaming  sme       p               chained     r"sel p\v\().b, p\fixed, p\v\().b, p\fixed\().b"                      "svsel_b";
    ST1B      "{z.s}, p, [x]"                              "u8"   "u32"  sve_streaming  sme       lines           throughput  r"st1b {{z\v\().s}}, p0, [x\v]"                                       "svst1b_u32";
    TRN2      "z.h, z.h, z.h"                              "u16"  "u16"  sve_streaming  sme       z               chained     r"trn2 z\v\().h, z\v\().h, z\fixed\().h"                              "svtrn2_u16";
    UABD      "z.b, p/m, z.b, z.b"                         "u8"   "u8"   sve_streaming  sme       z               chained     r"uabd z\v\().b, p0/m, z\v\().b, z\fixed\().b"                        "svabd_u8_x";
    UMAXV     "d, p, z.d"                                  "u64"  "u64"  sve_streaming  sme       z               chained     r"umaxv d\v, p0, z\v\().d"                                            "svmaxv_u64";
    UMINV     "b, p, z.b"                                  "u8"   "u8"   sve_streaming  sme       z               chained     r"uminv b\v, p0, z\v\().b"                                            "svminv_u8";
    ZIP1      "z.d, z.d, z.d"                              "u64"  "u64"  sve_streaming  sme       z               chained     r"zip1 z\v\().d, z\v\().d, z\fixed\().d"                              "svzip1_f64";
    ZIP1      "z.s, z.s, z.s"                              "u32"  "u32"  sve_streaming  sme       z               chained     r"zip1 z\v\().s, z\v\().s, z\fixed\().s"                              "svzip1_f32";
    ZIP2      "z.b, z.b, z.b"                              "u8"   "u8"   sve_streaming  sme       z               chained     r"zip2 z\v\().b, z\v\().b, z\fixed\().b"                              "svzip2_u8";
    ZIP2      "z.d, z.d, z.d"                              "u64"  "u64"  sve_streaming  sme       z               chained     r"zip2 z\v\().d, z\v\().d, z\fixed\().d"                              "svzip2_f64";

    FCVTLT    "z.s, p/m, z.h"                              "f16"  "f32"  sve2_streaming sme       z               chained     r"fcvtlt z\v\().s, p0/m, z\v\().h"                                    "svcvtlt_f32_f16_x";
    FCVTLT    "z.d, p/m, z.s"                              "f32"  "f64"  sve2_streaming sme       z               chained     r"fcvtlt z\v\().d, p0/m, z\v\().s"                                    "svcvtlt_f64_f32_x";
    TBL       "z.h, {z.h, z.h}, z.h"                       "u16"  "u16"  sve2_streaming sme       z_low           chained     r"tbl z\v\().h, {{z16.h, z17.h}}, z\v\().h"                           "svtbl2_u16";
    FCVTNT    "z.h, p/m, z.s"                              "f32"  "f16"  sve2_streaming sme       z               chained     r"fcvtnt z\v\().h, p0/m, z\fixed\().s"                                "svcvtnt_f16_f32_m";
    BSL       "z.d, z.d, z.d, z.d"                         "u64"  "u64"  sve2_streaming sme       z               chained     r"bsl z\v\().d, z\v\().d, z\fixed\().d, z\fixed\().d"                 "svbsl_n_u16";
    TBL       "z.b, {z.b, z.b}, z.b"                       "u8"   "u8"   sve2_streaming sme       z_low           chained     r"tbl z\v\().b, {{z16.b, z17.b}}, z\v\().b"                           "svtbl2_u8";
    UMULLB    "z.d, z.s, z.s"                              "u32"  "u64"  sve2_streaming sme       z               chained     r"umullb z\v\().d, z\v\().s, z\fixed\().s"                            "svmullb_n_u64 svmullb_u64";
    UMULLT    "z.d, z.s, z.s"                              "u32"  "u64"  sve2_streaming sme       z               chained     r"umullt z\v\().d, z\v\().s, z\fixed\().s"                            "svmullt_n_u64 svmullt_u64";
    UADDLB    "z.d, z.s, z.s"                              "u32"  "u64"  sve2_streaming sme       z               chained     r"uaddlb z\v\().d, z\v\().s, z\fixed\().s"                            "svaddlb_n_u64";
    UADDLT    "z.d, z.s, z.s"                              "u32"  "u64"  sve2_streaming sme       z               chained     r"uaddlt z\v\().d, z\v\().s, z\fixed\().s"                            "svaddlt_n_u64";
    SRSHR     "z.d, p/m, z.d, #imm"                        "i64"  "i64"  sve2_streaming sme       z               chained     r"srshr z\v\().d, p0/m, z\v\().d, #1"                                 "svrshr_n_s64_x";
    SMLSLB    "z.d, z.s, z.s"                              "i32"  "i64"  sve2_streaming sme       z               chained     r"smlslb z\v\().d, z\fixed\().s, z\fixed\().s"                        "svmlslb_n_s64";
    SMLSLT    "z.d, z.s, z.s"                              "i32"  "i64"  sve2_streaming sme       z               chained     r"smlslt z\v\().d, z\fixed\().s, z\fixed\().s"                        "svmlslt_n_s64";
    UMLSLB    "z.d, z.s, z.s"                              "u32"  "u64"  sve2_streaming sme       z               chained     r"umlslb z\v\().d, z\fixed\().s, z\fixed\().s"                        "svmlslb_n_u64";
    UMLSLT    "z.d, z.s, z.s"                              "u32"  "u64"  sve2_streaming sme       z               chained     r"umlslt z\v\().d, z\fixed\().s, z\fixed\().s"                        "svmlslt_n_u64";
    SMULLB    "z.d, z.s, z.s"                              "i32"  "i64"  sve2_streaming sme       z               chained     r"smullb z\v\().d, z\v\().s, z\fixed\().s"                            "svmullb_s64";
    SMULLT    "z.d, z.s, z.s"                              "i32"  "i64"  sve2_streaming sme       z               chained     r"smullt z\v\().d, z\v\().s, z\fixed\().s"                            "svmullt_s64";
    UADDWB    "z.h, z.h, z.b"                              "u8"   "u16"  sve2_streaming sme       z               chained     r"uaddwb z\v\().h, z\v\().h, z\fixed\().b"                            "svaddwb_u16";
    UADDWT    "z.h, z.h, z.b"                              "u8"   "u16"  sve2_streaming sme       z               chained     r"uaddwt z\v\().h, z\v\().h, z\fixed\().b"                            "svaddwt_u16";

    ZERO      "{za.d}"                                     ""     ""     sme            sme       za_d            throughput  r"zero {{\v\().d}}"                                                   "svzero_mask_za";
    MOV       "z.s, p/m, zav.s[w, imm]"                    "u32"  "u32"  sme            sme       z               throughput  r"mov z\v\().s, p0/m, za0v.s[w12, 0]"                                 "svread_ver_za32_f32_m svread_ver_za32_s32_m svread_ver_za32_u32_m";
    MOV       "zah.s[w, imm], p/m, z.s"                    "u32"  "u32"  sme            sme       za_s_horizontal throughput  r"mov \v, p0/m, \fixed\().s"                                          "svwrite_hor_za32_f32_m svwrite_hor_za32_s32_m svwrite_hor_za32_u32_m";
    FMOPA     "za.s, p/m, p/m, z.h, z.h"                   "f16"  "f32"  sme            sme       za_s            chained     r"fmopa \v\().s, p0/m, p0/m, \fixed\().h, \fixed\().h"                "svmopa_za32_f16_m";
    MOV       "z.d, p/m, zav.d[w, imm]"                    "u64"  "u64"  sme            sme       z               throughput  r"mov z\v\().d, p0/m, za0v.d[w12, 0]"                                 "svread_ver_za64_f64_m";
    MOV       "z.s, p/m, zah.s[w, imm]"                    "u32"  "u32"  sme            sme       z               throughput  r"mov z\v\().s, p0/m, za0h.s[w12, 0]"                                 "svread_hor_za32_f32_m svread_hor_za32_s32_m svread_hor_za32_u32_m";
    MOV       "z.d, p/m, zah.d[w, imm]"                    "u64"  "u64"  sme            sme       z               throughput  r"mov z\v\().d, p0/m, za0h.d[w12, 0]"                                 "svread_hor_za64_f64_m";
    BFMOPA    "za.s, p/m, p/m, z.h, z.h"                   "bf16" "f32"  sme            sme       za_s            chained     r"bfmopa \v\().s, p0/m, p0/m, \fixed\().h, \fixed\().h"               "svmopa_za32_bf16_m";
    ZERO      "{za}"                                       ""     ""     sme            sme       za_d            throughput  r"zero {{za}}"                                                        "svzero_za";
    SMOPA     "za.s, p/m, p/m, z.b, z.b"                   "i8"   "i32"  sme            sme       za_s            chained     r"smopa \v\().s, p0/m, p0/m, \fixed\().b, \fixed\().b"                "svmopa_za32_s8_m";
    MOV       "zah.d[w, imm], p/m, z.d"                    "u64"  "u64"  sme            sme       za_d_horizontal throughput  r"mov \v, p0/m, \fixed\().d"                                          "svwrite_hor_za64_f64_m";
    UMOPA     "za.s, p/m, p/m, z.b, z.b"                   "u8"   "u32"  sme            sme       za_s            chained     r"umopa \v\().s, p0/m, p0/m, \fixed\().b, \fixed\().b"                "svmopa_za32_u8_m";
    LD1W      "{zah.s[w, imm]}, p/z, [x]"                  "u32"  "u32"  sme            sme       za_s_horizontal throughput  r"ld1w {{\v}}, p0/z, [x8]"                                            "svld1_hor_za32";
    MOV       "zav.s[w, imm], p/m, z.s"                    "u32"  "u32"  sme            sme       za_s_vertical   throughput  r"mov \v, p0/m, \fixed\().s"                                          "svwrite_ver_za32_f32_m svwrite_ver_za32_s32_m svwrite_ver_za32_u32_m";
    LD1D      "{zah.d[w, imm]}, p/z, [x]"                  "u64"  "u64"  sme            sme       za_d_horizontal throughput  r"ld1d {{\v}}, p0/z, [x8]"                                            "svld1_hor_za64";
    MOV       "zav.d[w, imm], p/m, z.d"                    "u64"  "u64"  sme            sme       za_d_vertical   throughput  r"mov \v, p0/m, \fixed\().d"                                          "svwrite_ver_za64_f64_m";
    ST1W      "{zah.s[w, imm]}, p, [x]"                    "u32"  "u32"  sme            sme       za_s_horizontal throughput  r"st1w {{\v}}, p0, [x8]"                                              "svst1_hor_za32";
    ST1W      "{zav.s[w, imm]}, p, [x]"                    "u32"  "u32"  sme            sme       za_s_vertical   throughput  r"st1w {{\v}}, p0, [x8]"                                              "svst1_ver_za32";
    USMOPA    "za.s, p/m, p/m, z.b, z.b"                   "u8"   "i32"  sme            sme       za_s            chained     r"usmopa \v\().s, p0/m, p0/m, \fixed\().b, \fixed\().b"               "svusmopa_za32_u8_m";
    MOV       "zah.h[w, imm], p/m, z.h"                    "u16"  "u16"  sme            sme       za_h_horizontal throughput  r"mov \v, p0/m, \fixed\().h"                                          "svwrite_hor_za16_u16_m";
    RDSVL     "x, #imm"                                    "u64"  "u64"  sme            sme       x               throughput  r"rdsvl x\v, #1"                                                      "";
    ST1H      "{zav.h[w, imm]}, p, [x]"                    "u16"  "u16"  sme            sme       za_h_vertical   throughput  r"st1h {{\v}}, p0, [x8]"                                              "svst1_ver_za16";
    FMOPA     "za.s, p/m, p/m, z.s, z.s"                   "f32"  "f32"  sme            sme       za_s            chained     r"fmopa \v\().s, p0/m, p0/m, \fixed\().s, \fixed\().s"                "";

    BMOPA     "za.s, p/m, p/m, z.s, z.s"                   "u1"   "u32"  sme2           sme2      za_s            chained     r"bmopa \v\().s, p0/m, p0/m, \fixed\().s, \fixed\().s"                "svbmopa_za32_u32_m";
    SMOPA     "za.s, p/m, p/m, z.h, z.h"                   "i16"  "i32"  sme2           sme2      za_s            chained     r"smopa \v\().s, p0/m, p0/m, \fixed\().h, \fixed\().h"                "";
    LUTI4     "z.b, zt0, z[0]"                             "u4"   "u8"   sme2           sme2      z               independent r"luti4 z\v\().b, zt0, z\fixed[0]"                                    "";
    LUTI2     "z.b, zt0, z[0]"                             "u2"   "u8"   sme2           sme2      z               independent r"luti2 z\v\().b, zt0, z\fixed[0]"                                    "";
    UDOT      "za.s[w, 0, vgx4], {z.b-z.b}, z.b"           "u8"   "u32"  sme2           sme2      vgx4            chained     r"udot za.s[w11, \v, vgx4], {{z0.b-z3.b}}, \fixed\().b"               "";
    FDOT      "za.s[w, 0, vgx4], {z.h-z.h}, z.h"           "f16"  "f32"  sme2           sme2      vgx4            chained     r"fdot za.s[w11, \v, vgx4], {{z0.h-z3.h}}, \fixed\().h"               "";
    FMLA      "za.s[w, 0, vgx4], {z.s-z.s}, z.s"           "f32"  "f32"  sme2           sme2      vgx4            chained     r"fmla za.s[w11, \v, vgx4], {{z0.s-z3.s}}, \fixed\().s"               "";

    FMOPA     "za.h, p/m, p/m, z.h, z.h"                   "f16"  "f16"  sme_f16f16     smef16f16 za_h            chained     r"fmopa \v\().h, p0/m, p0/m, \fixed\().h, \fixed\().h"                "";

    FMOPA     "za.d, p/m, p/m, z.d, z.d"                   "f64"  "f64"  sme_f64f64     smef64f64 za_d            chained     r"fmopa \v\().d, p0/m, p0/m, \fixed\().d, \fixed\().d"                "svmopa_za64_f64_m";
    FMOPS     "za.d, p/m, p/m, z.d, z.d"                   "f64"  "f64"  sme_f64f64     smef64f64 za_d            chained     r"fmops \v\().d, p0/m, p0/m, \fixed\().d, \fixed\().d"                "svmops_za64_f64_m";
}

/// Forms no single step template can time: round trips through another register file or mode,
/// exclusive pairs, and register-pair atomics, each a latency chain of its whole sequence.
pub const EXCEPTIONS: &[Form] = &[
    Form {
        mnemonic: "MOVA",
        form: "z.s -> za0h.s[w, 0] -> z.s",
        input_type: "u32",
        output_type: "u32",
        extension: "sme",
        feature: "sme",
        mode: Mode::Streaming,
        reference: &STREAMING_REFERENCE,
        latency: Some(kernels!(
            sme,
            "mova za0h.s[w12, 0], p0/m, z\\v\\().s\nmova z\\v\\().s, p0/m, za0h.s[w12, 0]",
            class!(z, first),
            class!(z, fixed)
        )),
        throughput: None,
        intrinsics: "",
    },
    Form {
        mnemonic: "SMSTART",
        form: "then SMSTOP",
        input_type: "",
        output_type: "",
        extension: "sme",
        feature: "sme",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(neon, "smstart\nsmstop", "x11", "x11")),
        throughput: None,
        intrinsics: "",
    },
    Form {
        mnemonic: "SMSTART",
        form: "sm, then SMSTOP sm",
        input_type: "",
        output_type: "",
        extension: "sme",
        feature: "sme",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(neon, "smstart sm\nsmstop sm", "x11", "x11")),
        throughput: None,
        intrinsics: "",
    },
    Form {
        mnemonic: "FMOV",
        form: "d, x -> x, d",
        input_type: "u64",
        output_type: "u64",
        extension: "neon",
        feature: "asimd",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(
            neon,
            "fmov d\\v, x\\v\nfmov x\\v, d\\v",
            class!(x, first),
            class!(x, fixed)
        )),
        throughput: None,
        intrinsics: "",
    },
    // The loaded line holds its own address, so moving it back to x\v chases it.
    Form {
        mnemonic: "LDR",
        form: "q, [x] -> FMOV x, d",
        input_type: "u128",
        output_type: "u128",
        extension: "neon",
        feature: "asimd",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(
            neon,
            "ldr q\\v, [x\\v]\nfmov x\\v, d\\v",
            class!(lines, first),
            class!(lines, fixed)
        )),
        throughput: None,
        intrinsics: "",
    },
    Form {
        mnemonic: "LDXR",
        form: "x, [x] -> STXR w, x, [x]",
        input_type: "u64",
        output_type: "u64",
        extension: "atomics",
        feature: "fp",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(
            atomics,
            "ldxr x\\v, [x\\v]\nstxr w1\\v, x\\v, [x\\v]",
            class!(lines, first),
            class!(lines, fixed)
        )),
        throughput: Some(kernels!(
            atomics,
            "ldxr x\\v, [x\\v]\nstxr w1\\v, x\\v, [x\\v]",
            class!(lines, rotated),
            class!(lines, fixed)
        )),
        intrinsics: "",
    },
    Form {
        mnemonic: "LDAXR",
        form: "x, [x] -> STLXR w, x, [x]",
        input_type: "u64",
        output_type: "u64",
        extension: "atomics",
        feature: "fp",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(
            atomics,
            "ldaxr x\\v, [x\\v]\nstlxr w1\\v, x\\v, [x\\v]",
            class!(lines, first),
            class!(lines, fixed)
        )),
        throughput: Some(kernels!(
            atomics,
            "ldaxr x\\v, [x\\v]\nstlxr w1\\v, x\\v, [x\\v]",
            class!(lines, rotated),
            class!(lines, fixed)
        )),
        intrinsics: "",
    },
    // SEVL sets the event register, so WFE consumes it and falls through without waiting.
    Form {
        mnemonic: "SEVL",
        form: "then WFE",
        input_type: "",
        output_type: "",
        extension: "atomics",
        feature: "fp",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(atomics, "sevl\nwfe", "x11", "x11")),
        throughput: None,
        intrinsics: "",
    },
    // The register pairs are compared and swapped in, so every swap after the first succeeds.
    Form {
        mnemonic: "CASP",
        form: "x, x, x, x, [x]",
        input_type: "u128",
        output_type: "u128",
        extension: "lse",
        feature: "atomics",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(
            lse,
            "casp x10, x11, x10, x11, [x\\v]",
            class!(lines, first),
            class!(lines, fixed)
        )),
        throughput: None,
        intrinsics: "",
    },
    Form {
        mnemonic: "CASPAL",
        form: "x, x, x, x, [x]",
        input_type: "u128",
        output_type: "u128",
        extension: "lse",
        feature: "atomics",
        mode: Mode::Standard,
        reference: &CORE_REFERENCE,
        latency: Some(kernels!(
            lse,
            "caspal x10, x11, x10, x11, [x\\v]",
            class!(lines, first),
            class!(lines, fixed)
        )),
        throughput: None,
        intrinsics: "",
    },
];
