//! SSE, AVX2, AVX-512, AMX, atomic and scalar forms as assembly text.
//!
//! Throughput kernels rotate over 15 VEX-encodable or 31 EVEX-encodable vector registers, 6 mask
//! registers, 6 general registers, 6 tile accumulators or 6 cache lines.
//!
//! See also:
//! - <https://github.com/ashvardanian/less_slow.cpp/blob/main/less_slow_amd64.S>
//! - <https://ashvardanian.com/posts/cpu-ports/>
//! - <https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/>
//! - <https://ashvardanian.com/posts/search-utf8/>
//! - <https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/>

use std::arch::asm;
use std::arch::x86_64::__cpuid_count;
use std::sync::OnceLock;

use crate::{Form, Reference};

pub const NAME: &str = "x86";

/// Register classes: the latency kernel's register, the throughput kernel's rotation, and the
/// reserved operands, the highest-numbered registers the rotation leaves out.
///
/// Vector and mask classes are register numbers, so one template spells the widths it mixes, as in
/// `xmm\v, ymm\v`. Masked forms write under `k1` and general-register operands of vector forms are
/// `eax` or `rax`, which the vector prologues fill.
macro_rules! class {
    (vex, $part:ident) => { class!(@ $part, "0", "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14", "15") };
    (evex, $part:ident) => { class!(@ $part, "0",
        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, \
         29, 30", "31") };
    (k, $part:ident) => { class!(@ $part, "1", "1, 2, 3, 4, 5, 6", "7") };
    // Kernels count loops in r11 and point r10 at the lines, so neither rotates.
    (r64, $part:ident) => { class!(@ $part, "rax", "rax, rcx, rsi, rdi, r8, r9", "rdx") };
    (r32, $part:ident) => { class!(@ $part, "eax", "eax, ecx, esi, edi, r8d, r9d", "edx") };
    // Tile instructions fault unless all three tiles differ, so two sources stay reserved.
    (tmm, $part:ident) => { class!(@ $part, "tmm0", "tmm0, tmm1, tmm2, tmm3, tmm4, tmm5", "tmm6, tmm7") };
    // Line addresses ride in the argument registers; each line holds its own address.
    (lines, $part:ident) => { class!(@ $part, "rdi", "rdi, rsi, rdx, rcx, r8, r9", "rax") };
    (@ first, $first:literal, $rotated:literal, $fixed:literal) => { $first };
    (@ rotated, $first:literal, $rotated:literal, $fixed:literal) => { $rotated };
    (@ fixed, $first:literal, $rotated:literal, $fixed:literal) => { $fixed };
}

/// Extensions: the CSV stem, the mode, the reference, and the code that fills
/// every register of the classes with `0x3f` bytes, finite and normal in every type, and leaves it.
///
/// The SSE prologue stays legacy-encoded, so the upper halves it leaves clean cost legacy forms no
/// merge with dirty YMM state.
macro_rules! extension {
    (core, $part:ident) => { extension!(@ $part, "", (crate::Mode::Standard), (&CORE_REFERENCE), "", "") };
    (sse, $part:ident) => { extension!(@ $part, "sse", (crate::Mode::Standard), (&CORE_REFERENCE),
        concat!("mov eax, 0x3f3f3f3f\nmovd xmm0, eax\npshufd xmm0, xmm0, 0\n.irp register, ",
                class!(vex, rotated), ", ", class!(vex, fixed), "\nmovdqa xmm\\register, xmm0\n.endr"), "") };
    (avx2, $part:ident) => { extension!(@ $part, "avx2", (crate::Mode::Standard), (&CORE_REFERENCE),
        concat!("mov eax, 0x3f3f3f3f\nvmovd xmm0, eax\nvpbroadcastd ymm0, xmm0\n.irp register, ",
                class!(vex, rotated), ", ", class!(vex, fixed), "\nvmovdqa ymm\\register, ymm0\n.endr"),
        "vzeroupper") };
    (avx512, $part:ident) => { extension!(@ $part, "avx512", (crate::Mode::Standard), (&CORE_REFERENCE),
        concat!("mov rax, 0x3f3f3f3f3f3f3f3f\nvpbroadcastq zmm0, rax\n.irp register, ",
                class!(evex, rotated), ", ", class!(evex, fixed), "\nvmovdqa64 zmm\\register, zmm0\n.endr\n",
                ".irp register, ", class!(k, rotated), ", ", class!(k, fixed), "\nkmovq k\\register, rax\n.endr"),
        "vzeroupper") };
    (amx, $part:ident) => { extension!(@ $part, "amx", (crate::Mode::Standard), (&CORE_REFERENCE),
        concat!("ldtilecfg [rax]\n.irp register, ", class!(tmm, rotated), ", ", class!(tmm, fixed),
                "\ntilezero \\register\n.endr"),
        "tilerelease") };
    (atomics, $part:ident) => { extension!(@ $part, "atomics", (crate::Mode::Standard), (&CORE_REFERENCE), "", "") };
    (scalar, $part:ident) => { extension!(@ $part, "scalar", (crate::Mode::Standard), (&CORE_REFERENCE),
        concat!("mov rax, 0x3f3f3f3f3f3f3f3f\n.irp register, ", class!(r64, rotated), ", ", class!(r64, fixed),
                "\nmov \\register, rax\n.endr"), "") };
    (@ stem, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $stem };
    (@ mode, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $mode };
    (@ reference, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $reference };
    (@ prologue, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $prologue };
    (@ epilogue, $stem:literal, $mode:tt, $reference:tt, $prologue:expr, $epilogue:literal) => { $epilogue };
}

/// The `UNROLL` and `2 × UNROLL` kernels of a step template, each copy taking the next entry of
/// `registers` as `\v` and `fixed` as `\fixed`.
///
/// Every kernel points the `lines` registers at lines of its own stack before the prologue, so
/// loads, stores and atomics of any extension stay in L1.
macro_rules! kernels {
    (amx, $template:literal, $registers:expr, $fixed:expr) => {{
        fn kernel<const COPIES: usize>(iterations: u64) {
            let mut lines = [crate::Line::default(); crate::registers(class!(lines, rotated))];
            let configuration = tile_configuration();
            unsafe {
                kernels!(@asm amx, $template, $registers, $fixed,
                    line_bytes = const size_of::<crate::Line>(),
                    rotations = const COPIES / crate::registers($registers),
                    in("rax") configuration,
                    inout("r10") lines.as_mut_ptr() => _,
                    inout("r11") iterations => _,
                    clobber_abi("C"),
                    options(nostack),
                )
            }
        }
        [kernel::<UNROLL> as crate::Kernel, kernel::<{ 2 * UNROLL }>]
    }};
    ($extension:ident, $template:literal, $registers:expr, $fixed:expr) => {{
        fn kernel<const COPIES: usize>(iterations: u64) {
            let mut lines = [crate::Line::default(); crate::registers(class!(lines, rotated))];
            unsafe {
                kernels!(@asm $extension, $template, $registers, $fixed,
                    line_bytes = const size_of::<crate::Line>(),
                    rotations = const COPIES / crate::registers($registers),
                    inout("r10") lines.as_mut_ptr() => _,
                    inout("r11") iterations => _,
                    clobber_abi("C"),
                    options(nostack),
                )
            }
        }
        [kernel::<UNROLL> as crate::Kernel, kernel::<{ 2 * UNROLL }>]
    }};
    (@asm $extension:ident, $template:literal, $registers:expr, $fixed:expr, $($operands:tt)*) => {
        asm!(
            ".macro step v, fixed",
            $template,
            ".endm",
            ".set line_offset, 0",
            concat!(".irp register, ", class!(lines, rotated)),
            "lea \\register, [r10 + line_offset]",
            "mov qword ptr [\\register], \\register",
            ".set line_offset, line_offset + {line_bytes}",
            ".endr",
            extension!($extension, prologue),
            "2:",
            ".rept {rotations}",
            concat!(".irp register, ", $registers),
            concat!(r#"step "\register", ""#, $fixed, "\""),
            ".endr",
            ".endr",
            "dec r11",
            "jnz 2b",
            extension!($extension, epilogue),
            ".purgem step",
            $($operands)*
        )
    };
}

/// The palette 1 layout `LDTILECFG` reads.
#[derive(Default)]
#[repr(C, align(64))]
struct TileConfiguration {
    palette: u8,
    start_row: u8,
    reserved: [u8; 14],
    bytes_per_row: [u16; 16],
    rows: [u8; 16],
}

/// Linux `arch_prctl` request and feature number for the AMX tile state, from `asm/prctl.h`.
const ARCH_REQ_XCOMP_PERM: libc::c_long = 0x1023;
const XFEATURE_XTILEDATA: libc::c_long = 18;
/// CPUID leaf describing tile palettes, and the palette the kernels load.
const TILE_INFORMATION_LEAF: u32 = 0x1D;
const PALETTE: u32 = 1;

/// Asks Linux for the tile state once, then configures every tile at the largest shape CPUID
/// reports for palette 1.
fn tile_configuration() -> *const TileConfiguration {
    static CONFIGURATION: OnceLock<TileConfiguration> = OnceLock::new();
    CONFIGURATION.get_or_init(|| {
        unsafe {
            libc::syscall(
                libc::SYS_arch_prctl,
                ARCH_REQ_XCOMP_PERM,
                XFEATURE_XTILEDATA,
            )
        };
        let palette = __cpuid_count(TILE_INFORMATION_LEAF, PALETTE);
        let names = (palette.ebx >> u16::BITS) as usize;
        let mut configuration = TileConfiguration {
            palette: PALETTE as u8,
            ..Default::default()
        };
        for tile in 0..names.min(configuration.rows.len()) {
            configuration.bytes_per_row[tile] = palette.ebx as u16;
            configuration.rows[tile] = palette.ecx as u8;
        }
        configuration
    })
}

// Golden Cove and later fold dependent adds of an immediate at rename, so the chain adds a register.
pub static CORE_REFERENCE: Reference = Reference {
    name: "add r64",
    kernels: kernels!(core, r"add \v, \v", "r10", "r10"),
};

forms! {
//  mnemonic         form                    input_type output_type extension feature             class kernels    template                                          intrinsics
    PREFETCHT1       "m8"                    ""     ""     sse     sse                 lines throughput r"prefetcht1 byte ptr [\v]"                       "";
    PREFETCHT2       "m8"                    ""     ""     sse     sse                 lines throughput r"prefetcht2 byte ptr [\v]"                       "_mm_prefetch";
    SHA256MSG1       "xmm, xmm"              "u32"  "u32"  sse     sha_ni              vex   chained    r"sha256msg1 xmm\v, xmm\fixed"                    "_mm_sha256msg1_epu32";
    SHA256MSG2       "xmm, xmm"              "u32"  "u32"  sse     sha_ni              vex   chained    r"sha256msg2 xmm\v, xmm\fixed"                    "_mm_sha256msg2_epu32";
    SHA256RNDS2      "xmm, xmm, xmm"         "u32"  "u32"  sse     sha_ni              vex   chained    r"sha256rnds2 xmm\v, xmm\fixed, xmm0"             "_mm_sha256rnds2_epu32";
    VADDPD           "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vaddpd xmm\v, xmm\v, xmm\fixed"                 "_mm_add_pd";
    VADDPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vaddps xmm\v, xmm\v, xmm\fixed"                 "_mm_add_ps";
    VADDSD           "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vaddsd xmm\v, xmm\v, xmm\fixed"                 "_mm_add_sd";
    VAESENC          "xmm, xmm, xmm"         "u128" "u128" sse     aes                 vex   chained    r"vaesenc xmm\v, xmm\v, xmm\fixed"                "_mm_aesenc_si128";
    VAESENCLAST      "xmm, xmm, xmm"         "u128" "u128" sse     aes                 vex   chained    r"vaesenclast xmm\v, xmm\v, xmm\fixed"            "_mm_aesenclast_si128";
    VAESKEYGENASSIST "xmm, xmm, imm8"        "u128" "u128" sse     aes                 vex   chained    r"vaeskeygenassist xmm\v, xmm\v, 1"               "_mm_aeskeygenassist_si128";
    VANDNPS          "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vandnps xmm\v, xmm\v, xmm\fixed"                "_mm_andnot_ps";
    VANDPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vandps xmm\v, xmm\v, xmm\fixed"                 "_mm_and_ps _mm_and_si128";
    VBLENDVPS        "xmm, xmm, xmm, xmm"    "f32"  "f32"  sse     avx                 vex   chained    r"vblendvps xmm\v, xmm\v, xmm\fixed, xmm\fixed"   "_mm_blendv_ps";
    VBROADCASTSS     "xmm, xmm"              "f32"  "f32"  sse     avx2                vex   chained    r"vbroadcastss xmm\v, xmm\v"                      "_mm_set1_ps";
    VCMPEQPS         "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vcmpeqps xmm\v, xmm\v, xmm\fixed"               "_mm_cmpeq_ps";
    VCMPNEQPS        "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vcmpneqps xmm\v, xmm\v, xmm\fixed"              "_mm_cmpneq_ps";
    VCMPUNORDPS      "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vcmpunordps xmm\v, xmm\v, xmm\fixed"            "_mm_cmpunord_ps";
    VCVTDQ2PS        "xmm, xmm"              "i32"  "f32"  sse     avx                 vex   chained    r"vcvtdq2ps xmm\v, xmm\v"                         "_mm_cvtepi32_ps";
    VDIVPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vdivps xmm\v, xmm\fixed, xmm\v"                 "_mm_div_ps";
    VEXTRACTPS       "r32, xmm, imm8"        "u32"  "u32"  sse     avx                 vex   throughput r"vextractps eax, xmm\v, 1"                       "_mm_extract_epi32";
    VHADDPD          "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vhaddpd xmm\v, xmm\v, xmm\fixed"                "_mm_hadd_pd";
    VHADDPS          "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vhaddps xmm\v, xmm\v, xmm\fixed"                "_mm_hadd_ps";
    VLDDQU           "xmm, m128"             "u8"   "u8"   sse     avx                 lines throughput r"vlddqu xmm0, xmmword ptr [\v]"                  "_mm_lddqu_si128";
    VMAXPD           "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vmaxpd xmm\v, xmm\v, xmm\fixed"                 "_mm_max_pd";
    VMAXPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vmaxps xmm\v, xmm\v, xmm\fixed"                 "_mm_max_ps";
    VMAXSS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vmaxss xmm\v, xmm\v, xmm\fixed"                 "_mm_max_ss";
    VMINPD           "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vminpd xmm\v, xmm\v, xmm\fixed"                 "_mm_min_pd";
    VMINPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vminps xmm\v, xmm\v, xmm\fixed"                 "_mm_min_ps";
    VMOVAPS          "m128, xmm"             "f32"  "f32"  sse     avx                 lines throughput r"vmovaps xmmword ptr [\v], xmm0"                 "_mm_store_si128";
    VMOVAPS          "xmm, m128"             "f32"  "f32"  sse     avx                 lines throughput r"vmovaps xmm0, xmmword ptr [\v]"                 "_mm_load_si128";
    VMOVD            "r32, xmm"              "u32"  "u32"  sse     avx                 vex   throughput r"vmovd eax, xmm\v"                               "_mm_cvtsi128_si32";
    VMOVD            "xmm, r32"              "u32"  "u32"  sse     avx                 vex   throughput r"vmovd xmm\v, eax"                               "_mm_cvtsi32_si128";
    VMOVLPS          "m64, xmm"              "u64"  "u64"  sse     avx                 lines throughput r"vmovlps qword ptr [\v], xmm0"                   "_mm_storel_epi64";
    VMOVQ            "r64, xmm"              "u64"  "u64"  sse     avx                 vex   throughput r"vmovq rax, xmm\v"                               "_mm_cvtsi128_si64";
    VMOVQ            "xmm, r64"              "u64"  "u64"  sse     avx                 vex   throughput r"vmovq xmm\v, rax"                               "_mm_cvtsi64_si128";
    VMOVQ            "xmm, xmm"              "u64"  "u64"  sse     avx                 vex   chained    r"vmovq xmm\v, xmm\v"                             "_mm_set_sd";
    VMOVSD           "xmm, m64"              "f64"  "f64"  sse     avx                 lines throughput r"vmovsd xmm0, qword ptr [\v]"                    "_mm_loadl_epi64";
    VMOVSS           "xmm, m32"              "f32"  "f32"  sse     avx                 lines throughput r"vmovss xmm0, dword ptr [\v]"                    "_mm_loadu_si32";
    VMOVSS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vmovss xmm\v, xmm\v, xmm\fixed"                 "_mm_blend_ps";
    VMOVUPS          "m128, xmm"             "f32"  "f32"  sse     avx                 lines throughput r"vmovups xmmword ptr [\v], xmm0"                 "_mm_storeu_epi32 _mm_storeu_ps _mm_storeu_si128";
    VMOVUPS          "xmm, m128"             "f32"  "f32"  sse     avx                 lines throughput r"vmovups xmm0, xmmword ptr [\v]"                 "_mm_loadu_epi8 _mm_loadu_epi32 _mm_loadu_ps _mm_loadu_si128";
    VMULPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vmulps xmm\v, xmm\v, xmm\v"                     "_mm_mul_ps";
    VMULSD           "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vmulsd xmm\v, xmm\v, xmm\v"                     "_mm_mul_sd";
    VMULSS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vmulss xmm\v, xmm\v, xmm\v"                     "_mm_mul_ss";
    VORPS            "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vorps xmm\v, xmm\v, xmm\fixed"                  "_mm_or_ps _mm_or_si128";
    VPACKSSDW        "xmm, xmm, xmm"         "i32"  "i16"  sse     avx                 vex   chained    r"vpackssdw xmm\v, xmm\v, xmm\fixed"              "_mm_packs_epi32";
    VPACKSSWB        "xmm, xmm, xmm"         "i16"  "i8"   sse     avx                 vex   chained    r"vpacksswb xmm\v, xmm\v, xmm\fixed"              "_mm_packs_epi16";
    VPACKUSDW        "xmm, xmm, xmm"         "i32"  "u16"  sse     avx                 vex   chained    r"vpackusdw xmm\v, xmm\v, xmm\fixed"              "_mm_packus_epi32";
    VPACKUSWB        "xmm, xmm, xmm"         "i16"  "u8"   sse     avx                 vex   chained    r"vpackuswb xmm\v, xmm\v, xmm\fixed"              "_mm_packus_epi16";
    VPADDB           "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpaddb xmm\v, xmm\v, xmm\fixed"                 "_mm_add_epi8";
    VPADDD           "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vpaddd xmm\v, xmm\v, xmm\fixed"                 "_mm_add_epi32";
    VPADDQ           "xmm, xmm, xmm"         "i64"  "i64"  sse     avx                 vex   chained    r"vpaddq xmm\v, xmm\v, xmm\fixed"                 "_mm_add_epi64";
    VPADDSB          "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpaddsb xmm\v, xmm\v, xmm\fixed"                "_mm_adds_epi8";
    VPADDSW          "xmm, xmm, xmm"         "i16"  "i16"  sse     avx                 vex   chained    r"vpaddsw xmm\v, xmm\v, xmm\fixed"                "_mm_adds_epi16";
    VPADDUSB         "xmm, xmm, xmm"         "u8"   "u8"   sse     avx                 vex   chained    r"vpaddusb xmm\v, xmm\v, xmm\fixed"               "_mm_adds_epu8";
    VPADDUSW         "xmm, xmm, xmm"         "u16"  "u16"  sse     avx                 vex   chained    r"vpaddusw xmm\v, xmm\v, xmm\fixed"               "_mm_adds_epu16";
    VPALIGNR         "xmm, xmm, xmm, imm8"   "u8"   "u8"   sse     avx                 vex   chained    r"vpalignr xmm\v, xmm\v, xmm\fixed, 1"            "_mm_alignr_epi8";
    VPBLENDVB        "xmm, xmm, xmm, xmm"    "u8"   "u8"   sse     avx                 vex   chained    r"vpblendvb xmm\v, xmm\v, xmm\fixed, xmm\fixed"   "_mm_blendv_epi8";
    VPCLMULQDQ       "xmm, xmm, xmm, imm8"   "u64"  "u128" sse     pclmulqdq           vex   chained    r"vpclmulqdq xmm\v, xmm\v, xmm\fixed, 0"          "_mm_clmulepi64_si128";
    VPCMPEQB         "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpcmpeqb xmm\v, xmm\v, xmm\fixed"               "_mm_cmpeq_epi8";
    VPCMPEQD         "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vpcmpeqd xmm\v, xmm\v, xmm\fixed"               "_mm_cmpeq_epi32";
    VPCMPEQW         "xmm, xmm, xmm"         "i16"  "i16"  sse     avx                 vex   chained    r"vpcmpeqw xmm\v, xmm\v, xmm\fixed"               "_mm_cmpeq_epi16";
    VPCMPGTB         "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpcmpgtb xmm\v, xmm\v, xmm\fixed"               "_mm_cmpgt_epi8";
    VPCMPGTD         "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vpcmpgtd xmm\v, xmm\v, xmm\fixed"               "_mm_cmpgt_epi32";
    VPCMPGTQ         "xmm, xmm, xmm"         "i64"  "i64"  sse     avx                 vex   chained    r"vpcmpgtq xmm\v, xmm\v, xmm\fixed"               "_mm_cmpgt_epi64";
    VPCMPGTW         "xmm, xmm, xmm"         "i16"  "i16"  sse     avx                 vex   chained    r"vpcmpgtw xmm\v, xmm\v, xmm\fixed"               "_mm_cmpgt_epi16";
    VPEXTRQ          "r64, xmm, imm8"        "u64"  "u64"  sse     avx                 vex   throughput r"vpextrq rax, xmm\v, 1"                          "_mm256_extract_epi64 _mm_extract_epi64";
    VPHADDD          "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vphaddd xmm\v, xmm\v, xmm\fixed"                "_mm_hadd_epi32";
    VPINSRB          "xmm, xmm, r32, imm8"   "u8"   "u8"   sse     avx                 vex   chained    r"vpinsrb xmm\v, xmm\v, eax, 1"                   "_mm_insert_epi8";
    VPINSRD          "xmm, xmm, r32, imm8"   "u32"  "u32"  sse     avx                 vex   chained    r"vpinsrd xmm\v, xmm\v, eax, 1"                   "_mm_insert_epi32";
    VPINSRQ          "xmm, xmm, r64, imm8"   "u64"  "u64"  sse     avx                 vex   chained    r"vpinsrq xmm\v, xmm\v, rax, 1"                   "_mm_insert_epi64";
    VPMADDUBSW       "xmm, xmm, xmm"         "u8"   "i16"  sse     avx                 vex   chained    r"vpmaddubsw xmm\v, xmm\v, xmm\fixed"             "_mm_maddubs_epi16";
    VPMAXSB          "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpmaxsb xmm\v, xmm\v, xmm\fixed"                "_mm_max_epi8";
    VPMAXSD          "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vpmaxsd xmm\v, xmm\v, xmm\fixed"                "_mm_max_epi32";
    VPMAXSW          "xmm, xmm, xmm"         "i16"  "i16"  sse     avx                 vex   chained    r"vpmaxsw xmm\v, xmm\v, xmm\fixed"                "_mm_max_epi16";
    VPMAXUB          "xmm, xmm, xmm"         "u8"   "u8"   sse     avx                 vex   chained    r"vpmaxub xmm\v, xmm\v, xmm\fixed"                "_mm_max_epu8";
    VPMAXUD          "xmm, xmm, xmm"         "u32"  "u32"  sse     avx                 vex   chained    r"vpmaxud xmm\v, xmm\v, xmm\fixed"                "_mm_max_epu32";
    VPMAXUW          "xmm, xmm, xmm"         "u16"  "u16"  sse     avx                 vex   chained    r"vpmaxuw xmm\v, xmm\v, xmm\fixed"                "_mm_max_epu16";
    VPMINSB          "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpminsb xmm\v, xmm\v, xmm\fixed"                "_mm_min_epi8";
    VPMINSD          "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vpminsd xmm\v, xmm\v, xmm\fixed"                "_mm_min_epi32";
    VPMINSW          "xmm, xmm, xmm"         "i16"  "i16"  sse     avx                 vex   chained    r"vpminsw xmm\v, xmm\v, xmm\fixed"                "_mm_min_epi16";
    VPMINUB          "xmm, xmm, xmm"         "u8"   "u8"   sse     avx                 vex   chained    r"vpminub xmm\v, xmm\v, xmm\fixed"                "_mm_min_epu8";
    VPMINUD          "xmm, xmm, xmm"         "u32"  "u32"  sse     avx                 vex   chained    r"vpminud xmm\v, xmm\v, xmm\fixed"                "_mm_min_epu32";
    VPMINUW          "xmm, xmm, xmm"         "u16"  "u16"  sse     avx                 vex   chained    r"vpminuw xmm\v, xmm\v, xmm\fixed"                "_mm_min_epu16";
    VPMOVMSKB        "r32, xmm"              "u8"   "u8"   sse     avx                 vex   throughput r"vpmovmskb eax, xmm\v"                           "_mm_movemask_epi8";
    VPMOVZXBW        "xmm, xmm"              "u8"   "u16"  sse     avx                 vex   chained    r"vpmovzxbw xmm\v, xmm\v"                         "_mm_cvtepu8_epi16";
    VPSADBW          "xmm, xmm, xmm"         "u8"   "u16"  sse     avx                 vex   chained    r"vpsadbw xmm\v, xmm\v, xmm\fixed"                "_mm_sad_epu8";
    VPSHUFB          "xmm, xmm, xmm"         "u8"   "u8"   sse     avx                 vex   chained    r"vpshufb xmm\v, xmm\v, xmm\fixed"                "_mm_shuffle_epi8";
    VPSLLDQ          "xmm, xmm, imm8"        "u8"   "u8"   sse     avx                 vex   chained    r"vpslldq xmm\v, xmm\v, 1"                        "_mm_slli_si128";
    VPSRLDQ          "xmm, xmm, imm8"        "u8"   "u8"   sse     avx                 vex   chained    r"vpsrldq xmm\v, xmm\v, 1"                        "_mm_srli_si128";
    VPSUBB           "xmm, xmm, xmm"         "i8"   "i8"   sse     avx                 vex   chained    r"vpsubb xmm\v, xmm\v, xmm\fixed"                 "_mm_sub_epi8";
    VPSUBD           "xmm, xmm, xmm"         "i32"  "i32"  sse     avx                 vex   chained    r"vpsubd xmm\v, xmm\v, xmm\fixed"                 "_mm_sub_epi32";
    VPUNPCKHBW       "xmm, xmm, xmm"         "u8"   "u8"   sse     avx                 vex   chained    r"vpunpckhbw xmm\v, xmm\v, xmm\fixed"             "_mm_unpackhi_epi8";
    VPUNPCKHDQ       "xmm, xmm, xmm"         "u32"  "u32"  sse     avx                 vex   chained    r"vpunpckhdq xmm\v, xmm\v, xmm\fixed"             "_mm_unpackhi_ps _mm_unpackhi_epi32";
    VPUNPCKHQDQ      "xmm, xmm, xmm"         "u64"  "u64"  sse     avx                 vex   chained    r"vpunpckhqdq xmm\v, xmm\v, xmm\fixed"            "_mm_movehl_ps _mm_unpackhi_epi64 _mm_unpackhi_pd";
    VPUNPCKHWD       "xmm, xmm, xmm"         "u16"  "u16"  sse     avx                 vex   chained    r"vpunpckhwd xmm\v, xmm\v, xmm\fixed"             "_mm_unpackhi_epi16";
    VPUNPCKLBW       "xmm, xmm, xmm"         "u8"   "u8"   sse     avx                 vex   chained    r"vpunpcklbw xmm\v, xmm\v, xmm\fixed"             "_mm_unpacklo_epi8";
    VPUNPCKLDQ       "xmm, xmm, xmm"         "u32"  "u32"  sse     avx                 vex   chained    r"vpunpckldq xmm\v, xmm\v, xmm\fixed"             "_mm_unpacklo_ps _mm_unpacklo_epi32";
    VPUNPCKLQDQ      "xmm, xmm, xmm"         "u64"  "u64"  sse     avx                 vex   chained    r"vpunpcklqdq xmm\v, xmm\v, xmm\fixed"            "_mm_movelh_ps _mm_set_pd _mm_unpacklo_epi64";
    VPUNPCKLWD       "xmm, xmm, xmm"         "u16"  "u16"  sse     avx                 vex   chained    r"vpunpcklwd xmm\v, xmm\v, xmm\fixed"             "_mm_unpacklo_epi16";
    VROUNDPS         "xmm, xmm, imm8"        "f32"  "f32"  sse     avx                 vex   chained    r"vroundps xmm\v, xmm\v, 0"                       "_mm_floor_ps";
    VRSQRTPS         "xmm, xmm"              "f32"  "f32"  sse     avx                 vex   chained    r"vrsqrtps xmm\v, xmm\v"                          "_mm_rsqrt_ps";
    VRSQRTSS         "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vrsqrtss xmm\v, xmm\fixed, xmm\v"               "_mm_rsqrt_ss";
    VSHUFPD          "xmm, xmm, xmm, imm8"   "f64"  "f64"  sse     avx                 vex   chained    r"vshufpd xmm\v, xmm\v, xmm\fixed, 1"             "_mm_shuffle_pd";
    VSHUFPS          "xmm, xmm, xmm, imm8"   "f32"  "f32"  sse     avx                 vex   chained    r"vshufps xmm\v, xmm\v, xmm\fixed, 1"             "_mm_permute_ps _mm_shuffle_ps _mm_shuffle_epi32";
    VSQRTPD          "xmm, xmm"              "f64"  "f64"  sse     avx                 vex   chained    r"vsqrtpd xmm\v, xmm\v"                           "_mm_sqrt_pd";
    VSQRTPS          "xmm, xmm"              "f32"  "f32"  sse     avx                 vex   chained    r"vsqrtps xmm\v, xmm\v"                           "_mm_sqrt_ps";
    VSUBPD           "xmm, xmm, xmm"         "f64"  "f64"  sse     avx                 vex   chained    r"vsubpd xmm\v, xmm\v, xmm\fixed"                 "_mm_sub_pd";
    VSUBPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vsubps xmm\v, xmm\v, xmm\fixed"                 "_mm_sub_ps";
    VSUBSS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vsubss xmm\v, xmm\v, xmm\fixed"                 "_mm_sub_ss";
    VXORPS           "xmm, xmm, xmm"         "f32"  "f32"  sse     avx                 vex   chained    r"vxorps xmm\v, xmm\v, xmm\fixed"                 "_mm256_setzero_pd _mm256_setzero_ps _mm256_setzero_si256 _mm512_setzero _mm512_setzero_pd _mm512_setzero_ps _mm512_setzero_si512 _mm_setzero_ph _mm512_setzero_ph _mm_setzero_ps _mm_setzero_si128 _mm_xor_si128";
    VADDPD           "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vaddpd ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_pd";
    VADDPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vaddps ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_ps";
    VAESENC          "ymm, ymm, ymm"         "u128" "u128" avx2    vaes                vex   chained    r"vaesenc ymm\v, ymm\v, ymm\fixed"                "_mm256_aesenc_epi128";
    VANDNPS          "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vandnps ymm\v, ymm\v, ymm\fixed"                "_mm256_andnot_pd _mm256_andnot_ps _mm256_andnot_si256";
    VANDPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vandps ymm\v, ymm\v, ymm\fixed"                 "_mm256_and_pd _mm256_and_ps _mm256_and_si256";
    VBLENDPS         "ymm, ymm, ymm, imm8"   "f32"  "f32"  avx2    avx                 vex   chained    r"vblendps ymm\v, ymm\v, ymm\fixed, 1"            "_mm256_blend_pd _mm256_blend_epi32";
    VBLENDVPD        "ymm, ymm, ymm, ymm"    "f64"  "f64"  avx2    avx                 vex   chained    r"vblendvpd ymm\v, ymm\v, ymm\fixed, ymm\fixed"   "_mm256_blendv_pd";
    VBLENDVPS        "ymm, ymm, ymm, ymm"    "f32"  "f32"  avx2    avx                 vex   chained    r"vblendvps ymm\v, ymm\v, ymm\fixed, ymm\fixed"   "_mm256_blendv_ps";
    VBROADCASTSD     "ymm, xmm"              "f64"  "f64"  avx2    avx2                vex   chained    r"vbroadcastsd ymm\v, xmm\v"                      "_mm256_set1_pd";
    VBROADCASTSS     "ymm, xmm"              "f32"  "f32"  avx2    avx2                vex   chained    r"vbroadcastss ymm\v, xmm\v"                      "_mm256_set1_ps _mm256_broadcastd_epi32 _mm256_broadcastss_ps";
    VCMPLTPD         "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vcmpltpd ymm\v, ymm\v, ymm\fixed"               "_mm256_cmp_pd";
    VCMPLTPS         "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vcmpltps ymm\v, ymm\v, ymm\fixed"               "_mm256_cmp_ps";
    VCVTDQ2PD        "ymm, xmm"              "i32"  "f64"  avx2    avx                 vex   chained    r"vcvtdq2pd ymm\v, xmm\v"                         "_mm256_cvtepi32_pd";
    VCVTDQ2PS        "ymm, ymm"              "i32"  "f32"  avx2    avx                 vex   chained    r"vcvtdq2ps ymm\v, ymm\v"                         "_mm256_cvtepi32_ps";
    VCVTPD2DQ        "xmm, ymm"              "f64"  "i32"  avx2    avx                 vex   chained    r"vcvtpd2dq xmm\v, ymm\v"                         "_mm256_cvtpd_epi32";
    VCVTPD2PS        "xmm, ymm"              "f64"  "f32"  avx2    avx                 vex   chained    r"vcvtpd2ps xmm\v, ymm\v"                         "_mm256_cvtpd_ps";
    VCVTPS2DQ        "ymm, ymm"              "f32"  "i32"  avx2    avx                 vex   chained    r"vcvtps2dq ymm\v, ymm\v"                         "_mm256_cvtps_epi32";
    VCVTPS2PD        "ymm, xmm"              "f32"  "f64"  avx2    avx                 vex   chained    r"vcvtps2pd ymm\v, xmm\v"                         "_mm256_cvtps_pd";
    VCVTPS2PH        "xmm, xmm, imm8"        "f32"  "f16"  avx2    f16c                vex   chained    r"vcvtps2ph xmm\v, xmm\v, 0"                      "_mm_cvtps_ph";
    VCVTPS2PH        "xmm, ymm, imm8"        "f32"  "f16"  avx2    f16c                vex   chained    r"vcvtps2ph xmm\v, ymm\v, 0"                      "_mm256_cvtps_ph";
    VCVTTPD2DQ       "xmm, ymm"              "f64"  "i32"  avx2    avx                 vex   chained    r"vcvttpd2dq xmm\v, ymm\v"                        "_mm256_cvttpd_epi32";
    VCVTTPS2DQ       "ymm, ymm"              "f32"  "i32"  avx2    avx                 vex   chained    r"vcvttps2dq ymm\v, ymm\v"                        "_mm256_cvttps_epi32";
    VDIVPD           "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vdivpd ymm\v, ymm\fixed, ymm\v"                 "_mm256_div_pd";
    VDIVPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vdivps ymm\v, ymm\fixed, ymm\v"                 "_mm256_div_ps";
    VEXTRACTF128     "xmm, ymm, imm8"        "u128" "u128" avx2    avx                 vex   chained    r"vextractf128 xmm\v, ymm\v, 1"                   "_mm256_extractf128_pd _mm256_extractf128_ps _mm256_extracti128_si256 _mm512_extracti64x2_epi64 _mm512_extracti32x4_epi32";
    VFMADD213PD      "ymm, ymm, ymm"         "f64"  "f64"  avx2    fma                 vex   chained    r"vfmadd213pd ymm\v, ymm\fixed, ymm\fixed"        "_mm256_fmadd_pd";
    VFMADD213PS      "xmm, xmm, xmm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfmadd213ps xmm\v, xmm\fixed, xmm\fixed"        "_mm_fmadd_ps";
    VFMADD213PS      "ymm, ymm, ymm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfmadd213ps ymm\v, ymm\fixed, ymm\fixed"        "_mm256_fmadd_ps";
    VFMADD213SD      "xmm, xmm, xmm"         "f64"  "f64"  avx2    fma                 vex   chained    r"vfmadd213sd xmm\v, xmm\fixed, xmm\fixed"        "_mm_fmadd_sd";
    VFMADD213SS      "xmm, xmm, xmm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfmadd213ss xmm\v, xmm\fixed, xmm\fixed"        "_mm_fmadd_ss";
    VFMADDSUB213PD   "ymm, ymm, ymm"         "f64"  "f64"  avx2    fma                 vex   chained    r"vfmaddsub213pd ymm\v, ymm\fixed, ymm\fixed"     "_mm256_fmaddsub_pd";
    VFMADDSUB213PS   "ymm, ymm, ymm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfmaddsub213ps ymm\v, ymm\fixed, ymm\fixed"     "_mm256_fmaddsub_ps";
    VFMSUB213PD      "ymm, ymm, ymm"         "f64"  "f64"  avx2    fma                 vex   chained    r"vfmsub213pd ymm\v, ymm\fixed, ymm\fixed"        "_mm256_fmsub_pd";
    VFMSUB213PS      "ymm, ymm, ymm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfmsub213ps ymm\v, ymm\fixed, ymm\fixed"        "_mm256_fmsub_ps";
    VFNMADD213PD     "ymm, ymm, ymm"         "f64"  "f64"  avx2    fma                 vex   chained    r"vfnmadd213pd ymm\v, ymm\fixed, ymm\fixed"       "_mm256_fnmadd_pd";
    VFNMADD213PS     "xmm, xmm, xmm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfnmadd213ps xmm\v, xmm\fixed, xmm\fixed"       "_mm_fnmadd_ps";
    VFNMADD213PS     "ymm, ymm, ymm"         "f32"  "f32"  avx2    fma                 vex   chained    r"vfnmadd213ps ymm\v, ymm\fixed, ymm\fixed"       "_mm256_fnmadd_ps";
    VFNMADD213SD     "xmm, xmm, xmm"         "f64"  "f64"  avx2    fma                 vex   chained    r"vfnmadd213sd xmm\v, xmm\fixed, xmm\fixed"       "_mm_fnmadd_sd";
    VINSERTF128      "ymm, ymm, xmm, imm8"   "u128" "u128" avx2    avx                 vex   chained    r"vinsertf128 ymm\v, ymm\v, xmm\fixed, 1"         "_mm256_set_m128d _mm256_set_m128i _mm256_setr_m128i _mm256_broadcastsi128_si256 _mm256_inserti128_si256";
    VLDDQU           "ymm, m256"             "u8"   "u8"   avx2    avx                 lines throughput r"vlddqu ymm0, ymmword ptr [\v]"                  "_mm256_lddqu_si256";
    VMASKMOVPD       "m256, ymm, ymm"        "f64"  "f64"  avx2    avx                 lines throughput r"vmaskmovpd ymmword ptr [\v], ymm0, ymm15"       "_mm256_maskstore_pd";
    VMASKMOVPD       "ymm, ymm, m256"        "f64"  "f64"  avx2    avx                 lines throughput r"vmaskmovpd ymm0, ymm15, ymmword ptr [\v]"       "_mm256_maskload_pd";
    VMASKMOVPS       "m128, xmm, xmm"        "f32"  "f32"  avx2    avx                 lines throughput r"vmaskmovps xmmword ptr [\v], xmm0, xmm15"       "_mm_maskstore_ps";
    VMASKMOVPS       "m256, ymm, ymm"        "f32"  "f32"  avx2    avx                 lines throughput r"vmaskmovps ymmword ptr [\v], ymm0, ymm15"       "_mm256_maskstore_ps";
    VMASKMOVPS       "xmm, xmm, m128"        "f32"  "f32"  avx2    avx                 lines throughput r"vmaskmovps xmm0, xmm15, xmmword ptr [\v]"       "_mm_maskload_ps";
    VMASKMOVPS       "ymm, ymm, m256"        "f32"  "f32"  avx2    avx                 lines throughput r"vmaskmovps ymm0, ymm15, ymmword ptr [\v]"       "_mm256_maskload_ps";
    VMAXPD           "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vmaxpd ymm\v, ymm\v, ymm\fixed"                 "_mm256_max_pd";
    VMAXPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vmaxps ymm\v, ymm\v, ymm\fixed"                 "_mm256_max_ps";
    VMINPD           "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vminpd ymm\v, ymm\v, ymm\fixed"                 "_mm256_min_pd";
    VMINPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vminps ymm\v, ymm\v, ymm\fixed"                 "_mm256_min_ps";
    VMOVAPS          "m256, ymm"             "f32"  "f32"  avx2    avx                 lines throughput r"vmovaps ymmword ptr [\v], ymm0"                 "_mm256_store_ps _mm256_store_si256";
    VMOVAPS          "ymm, m256"             "f32"  "f32"  avx2    avx                 lines throughput r"vmovaps ymm0, ymmword ptr [\v]"                 "_mm256_load_ps _mm256_load_si256";
    VMOVDDUP         "ymm, ymm"              "f64"  "f64"  avx2    avx                 vex   chained    r"vmovddup ymm\v, ymm\v"                          "_mm256_movedup_pd";
    VMOVMSKPD        "r32, ymm"              "f64"  "f64"  avx2    avx                 vex   throughput r"vmovmskpd eax, ymm\v"                           "_mm256_movemask_pd";
    VMOVMSKPS        "r32, ymm"              "f32"  "f32"  avx2    avx                 vex   throughput r"vmovmskps eax, ymm\v"                           "_mm256_movemask_ps";
    VMOVNTDQA        "ymm, m256"             "u8"   "u8"   avx2    avx2                lines throughput r"vmovntdqa ymm0, ymmword ptr [\v]"               "_mm256_stream_load_si256";
    VMOVSHDUP        "ymm, ymm"              "f32"  "f32"  avx2    avx                 vex   chained    r"vmovshdup ymm\v, ymm\v"                         "_mm256_movehdup_ps";
    VMOVSLDUP        "ymm, ymm"              "f32"  "f32"  avx2    avx                 vex   chained    r"vmovsldup ymm\v, ymm\v"                         "_mm256_moveldup_ps";
    VMOVUPS          "m256, ymm"             "f32"  "f32"  avx2    avx                 lines throughput r"vmovups ymmword ptr [\v], ymm0"                 "_mm256_storeu_pd _mm256_storeu_ps _mm256_storeu_si256";
    VMOVUPS          "ymm, m256"             "f32"  "f32"  avx2    avx                 lines throughput r"vmovups ymm0, ymmword ptr [\v]"                 "_mm256_loadu_pd _mm256_loadu_ps _mm256_loadu_si256 _mm256_loadu_epi16 _mm256_loadu_epi8";
    VMULPD           "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vmulpd ymm\v, ymm\v, ymm\v"                     "_mm256_mul_pd";
    VMULPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vmulps ymm\v, ymm\v, ymm\v"                     "_mm256_mul_ps";
    VORPS            "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vorps ymm\v, ymm\v, ymm\fixed"                  "_mm256_or_pd _mm256_or_ps _mm256_or_si256";
    VPABSB           "ymm, ymm"              "i8"   "i8"   avx2    avx2                vex   chained    r"vpabsb ymm\v, ymm\v"                            "_mm256_abs_epi8";
    VPACKSSDW        "ymm, ymm, ymm"         "i32"  "i16"  avx2    avx2                vex   chained    r"vpackssdw ymm\v, ymm\v, ymm\fixed"              "_mm256_packs_epi32";
    VPACKSSWB        "ymm, ymm, ymm"         "i16"  "i8"   avx2    avx2                vex   chained    r"vpacksswb ymm\v, ymm\v, ymm\fixed"              "_mm256_packs_epi16";
    VPACKUSDW        "ymm, ymm, ymm"         "i32"  "u16"  avx2    avx2                vex   chained    r"vpackusdw ymm\v, ymm\v, ymm\fixed"              "_mm256_packus_epi32";
    VPACKUSWB        "ymm, ymm, ymm"         "i16"  "u8"   avx2    avx2                vex   chained    r"vpackuswb ymm\v, ymm\v, ymm\fixed"              "_mm256_packus_epi16";
    VPADDB           "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpaddb ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_epi8";
    VPADDD           "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpaddd ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_epi32";
    VPADDQ           "ymm, ymm, ymm"         "i64"  "i64"  avx2    avx2                vex   chained    r"vpaddq ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_epi64";
    VPADDSB          "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpaddsb ymm\v, ymm\v, ymm\fixed"                "_mm256_adds_epi8";
    VPADDSW          "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpaddsw ymm\v, ymm\v, ymm\fixed"                "_mm256_adds_epi16";
    VPADDUSB         "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpaddusb ymm\v, ymm\v, ymm\fixed"               "_mm256_adds_epu8";
    VPADDUSW         "ymm, ymm, ymm"         "u16"  "u16"  avx2    avx2                vex   chained    r"vpaddusw ymm\v, ymm\v, ymm\fixed"               "_mm256_adds_epu16";
    VPADDW           "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpaddw ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_epi16";
    VPALIGNR         "ymm, ymm, ymm, imm8"   "u8"   "u8"   avx2    avx2                vex   chained    r"vpalignr ymm\v, ymm\v, ymm\fixed, 1"            "_mm256_alignr_epi8";
    VPBLENDVB        "ymm, ymm, ymm, ymm"    "u8"   "u8"   avx2    avx2                vex   chained    r"vpblendvb ymm\v, ymm\v, ymm\fixed, ymm\fixed"   "_mm256_blendv_epi8";
    VPCMPEQB         "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpcmpeqb ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpeq_epi8";
    VPCMPEQD         "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpcmpeqd ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpeq_epi32";
    VPCMPEQQ         "ymm, ymm, ymm"         "i64"  "i64"  avx2    avx2                vex   chained    r"vpcmpeqq ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpeq_epi64";
    VPCMPEQW         "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpcmpeqw ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpeq_epi16";
    VPCMPGTB         "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpcmpgtb ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpgt_epi8";
    VPCMPGTD         "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpcmpgtd ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpgt_epi32";
    VPCMPGTQ         "ymm, ymm, ymm"         "i64"  "i64"  avx2    avx2                vex   chained    r"vpcmpgtq ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpgt_epi64";
    VPCMPGTW         "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpcmpgtw ymm\v, ymm\v, ymm\fixed"               "_mm256_cmpgt_epi16";
    VPDPBUSD         "ymm, ymm, ymm"         "u8"   "i32"  avx2    avx_vnni            vex   chained    r"{{vex}} vpdpbusd ymm\v, ymm\fixed, ymm\fixed"   "_mm256_dpbusd_avx_epi32";
    VPDPWSSD         "ymm, ymm, ymm"         "i16"  "i32"  avx2    avx_vnni            vex   chained    r"{{vex}} vpdpwssd ymm\v, ymm\fixed, ymm\fixed"   "_mm256_dpwssd_avx_epi32";
    VPERMPD          "ymm, ymm, imm8"        "f64"  "f64"  avx2    avx2                vex   chained    r"vpermpd ymm\v, ymm\v, 1"                        "_mm256_permute2f128_pd _mm256_permute2f128_ps _mm256_permute2x128_si256 _mm256_permute4x64_epi64";
    VPERMPS          "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx2                vex   chained    r"vpermps ymm\v, ymm\v, ymm\fixed"                "_mm256_permutevar8x32_epi32 _mm256_permutevar8x32_ps";
    VPMADDUBSW       "ymm, ymm, ymm"         "u8"   "i16"  avx2    avx2                vex   chained    r"vpmaddubsw ymm\v, ymm\v, ymm\fixed"             "_mm256_maddubs_epi16";
    VPMADDWD         "ymm, ymm, ymm"         "i16"  "i32"  avx2    avx2                vex   chained    r"vpmaddwd ymm\v, ymm\v, ymm\fixed"               "_mm256_madd_epi16";
    VPMASKMOVD       "m256, ymm, ymm"        "i32"  "i32"  avx2    avx2                lines throughput r"vpmaskmovd ymmword ptr [\v], ymm0, ymm15"       "_mm256_maskstore_epi32";
    VPMASKMOVD       "ymm, ymm, m256"        "i32"  "i32"  avx2    avx2                lines throughput r"vpmaskmovd ymm0, ymm15, ymmword ptr [\v]"       "_mm256_maskload_epi32";
    VPMASKMOVQ       "m256, ymm, ymm"        "i64"  "i64"  avx2    avx2                lines throughput r"vpmaskmovq ymmword ptr [\v], ymm0, ymm15"       "_mm256_maskstore_epi64";
    VPMAXSB          "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpmaxsb ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epi8";
    VPMAXSD          "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpmaxsd ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epi32";
    VPMAXSW          "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpmaxsw ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epi16";
    VPMAXUB          "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpmaxub ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epu8";
    VPMAXUD          "ymm, ymm, ymm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpmaxud ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epu32";
    VPMAXUW          "ymm, ymm, ymm"         "u16"  "u16"  avx2    avx2                vex   chained    r"vpmaxuw ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epu16";
    VPMINSB          "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpminsb ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epi8";
    VPMINSD          "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpminsd ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epi32";
    VPMINSW          "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpminsw ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epi16";
    VPMINUB          "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpminub ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epu8";
    VPMINUD          "ymm, ymm, ymm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpminud ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epu32";
    VPMINUW          "ymm, ymm, ymm"         "u16"  "u16"  avx2    avx2                vex   chained    r"vpminuw ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epu16";
    VPMOVMSKB        "r32, ymm"              "u8"   "u8"   avx2    avx2                vex   throughput r"vpmovmskb eax, ymm\v"                           "_mm256_movemask_epi8";
    VPMOVSXBD        "ymm, xmm"              "i8"   "i32"  avx2    avx2                vex   chained    r"vpmovsxbd ymm\v, xmm\v"                         "_mm256_cvtepi8_epi32";
    VPMOVSXBW        "ymm, xmm"              "i8"   "i16"  avx2    avx2                vex   chained    r"vpmovsxbw ymm\v, xmm\v"                         "_mm256_cvtepi8_epi16";
    VPMOVSXDQ        "ymm, xmm"              "i32"  "i64"  avx2    avx2                vex   chained    r"vpmovsxdq ymm\v, xmm\v"                         "_mm256_cvtepi32_epi64";
    VPMOVSXWD        "ymm, xmm"              "i16"  "i32"  avx2    avx2                vex   chained    r"vpmovsxwd ymm\v, xmm\v"                         "_mm256_cvtepi16_epi32";
    VPMOVZXBD        "ymm, xmm"              "u8"   "u32"  avx2    avx2                vex   chained    r"vpmovzxbd ymm\v, xmm\v"                         "_mm256_cvtepu8_epi32";
    VPMOVZXBQ        "ymm, xmm"              "u8"   "u64"  avx2    avx2                vex   chained    r"vpmovzxbq ymm\v, xmm\v"                         "_mm256_cvtepu8_epi64";
    VPMOVZXBW        "ymm, xmm"              "u8"   "u16"  avx2    avx2                vex   chained    r"vpmovzxbw ymm\v, xmm\v"                         "_mm256_cvtepu8_epi16";
    VPMOVZXDQ        "ymm, xmm"              "u32"  "u64"  avx2    avx2                vex   chained    r"vpmovzxdq ymm\v, xmm\v"                         "_mm256_cvtepu32_epi64";
    VPMOVZXWD        "ymm, xmm"              "u16"  "u32"  avx2    avx2                vex   chained    r"vpmovzxwd ymm\v, xmm\v"                         "_mm256_cvtepu16_epi32";
    VPMULDQ          "ymm, ymm, ymm"         "i32"  "i64"  avx2    avx2                vex   chained    r"vpmuldq ymm\v, ymm\v, ymm\fixed"                "_mm256_mul_epi32";
    VPMULLD          "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpmulld ymm\v, ymm\v, ymm\fixed"                "_mm256_mullo_epi32";
    VPMULUDQ         "ymm, ymm, ymm"         "u32"  "u64"  avx2    avx2                vex   chained    r"vpmuludq ymm\v, ymm\v, ymm\fixed"               "_mm256_mul_epu32";
    VPSADBW          "ymm, ymm, ymm"         "u8"   "u16"  avx2    avx2                vex   chained    r"vpsadbw ymm\v, ymm\v, ymm\fixed"                "_mm256_sad_epu8";
    VPSHUFB          "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpshufb ymm\v, ymm\v, ymm\fixed"                "_mm256_shuffle_epi8";
    VPSIGNB          "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpsignb ymm\v, ymm\v, ymm\fixed"                "_mm256_sign_epi8";
    VPSIGNW          "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpsignw ymm\v, ymm\v, ymm\fixed"                "_mm256_sign_epi16";
    VPSLLVD          "ymm, ymm, ymm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpsllvd ymm\v, ymm\v, ymm\fixed"                "_mm256_sllv_epi32";
    VPSRAVD          "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpsravd ymm\v, ymm\v, ymm\fixed"                "_mm256_srav_epi32";
    VPSRLVD          "xmm, xmm, xmm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpsrlvd xmm\v, xmm\v, xmm\fixed"                "_mm_srlv_epi32";
    VPSRLVD          "ymm, ymm, ymm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpsrlvd ymm\v, ymm\v, ymm\fixed"                "_mm256_srlv_epi32";
    VPSRLVQ          "ymm, ymm, ymm"         "u64"  "u64"  avx2    avx2                vex   chained    r"vpsrlvq ymm\v, ymm\v, ymm\fixed"                "_mm256_srlv_epi64";
    VPSUBB           "ymm, ymm, ymm"         "i8"   "i8"   avx2    avx2                vex   chained    r"vpsubb ymm\v, ymm\v, ymm\fixed"                 "_mm256_sub_epi8";
    VPSUBD           "ymm, ymm, ymm"         "i32"  "i32"  avx2    avx2                vex   chained    r"vpsubd ymm\v, ymm\v, ymm\fixed"                 "_mm256_sub_epi32";
    VPSUBQ           "ymm, ymm, ymm"         "i64"  "i64"  avx2    avx2                vex   chained    r"vpsubq ymm\v, ymm\v, ymm\fixed"                 "_mm256_sub_epi64";
    VPSUBUSB         "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpsubusb ymm\v, ymm\v, ymm\fixed"               "_mm256_subs_epu8";
    VPSUBW           "ymm, ymm, ymm"         "i16"  "i16"  avx2    avx2                vex   chained    r"vpsubw ymm\v, ymm\v, ymm\fixed"                 "_mm256_sub_epi16";
    VPUNPCKHBW       "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpunpckhbw ymm\v, ymm\v, ymm\fixed"             "_mm256_unpackhi_epi8";
    VPUNPCKHDQ       "ymm, ymm, ymm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpunpckhdq ymm\v, ymm\v, ymm\fixed"             "_mm256_unpackhi_epi32";
    VPUNPCKHQDQ      "ymm, ymm, ymm"         "u64"  "u64"  avx2    avx2                vex   chained    r"vpunpckhqdq ymm\v, ymm\v, ymm\fixed"            "_mm256_unpackhi_pd _mm256_unpackhi_epi64";
    VPUNPCKLBW       "ymm, ymm, ymm"         "u8"   "u8"   avx2    avx2                vex   chained    r"vpunpcklbw ymm\v, ymm\v, ymm\fixed"             "_mm256_unpacklo_epi8";
    VPUNPCKLDQ       "ymm, ymm, ymm"         "u32"  "u32"  avx2    avx2                vex   chained    r"vpunpckldq ymm\v, ymm\v, ymm\fixed"             "_mm256_unpacklo_epi32";
    VPUNPCKLQDQ      "ymm, ymm, ymm"         "u64"  "u64"  avx2    avx2                vex   chained    r"vpunpcklqdq ymm\v, ymm\v, ymm\fixed"            "_mm256_unpacklo_pd _mm256_unpacklo_epi64";
    VROUNDPD         "ymm, ymm, imm8"        "f64"  "f64"  avx2    avx                 vex   chained    r"vroundpd ymm\v, ymm\v, 0"                       "_mm256_round_pd";
    VROUNDPS         "ymm, ymm, imm8"        "f32"  "f32"  avx2    avx                 vex   chained    r"vroundps ymm\v, ymm\v, 0"                       "_mm256_round_ps";
    VSHUFPD          "ymm, ymm, ymm, imm8"   "f64"  "f64"  avx2    avx                 vex   chained    r"vshufpd ymm\v, ymm\v, ymm\fixed, 1"             "_mm256_permute_pd _mm256_shuffle_pd";
    VSHUFPS          "ymm, ymm, ymm, imm8"   "f32"  "f32"  avx2    avx                 vex   chained    r"vshufps ymm\v, ymm\v, ymm\fixed, 1"             "_mm256_permute_ps _mm256_shuffle_ps _mm256_shuffle_epi32";
    VSQRTPD          "ymm, ymm"              "f64"  "f64"  avx2    avx                 vex   chained    r"vsqrtpd ymm\v, ymm\v"                           "_mm256_sqrt_pd";
    VSQRTPS          "ymm, ymm"              "f32"  "f32"  avx2    avx                 vex   chained    r"vsqrtps ymm\v, ymm\v"                           "_mm256_sqrt_ps";
    VSUBPD           "ymm, ymm, ymm"         "f64"  "f64"  avx2    avx                 vex   chained    r"vsubpd ymm\v, ymm\v, ymm\fixed"                 "_mm256_sub_pd";
    VSUBPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vsubps ymm\v, ymm\v, ymm\fixed"                 "_mm256_sub_ps";
    VXORPS           "ymm, ymm, ymm"         "f32"  "f32"  avx2    avx                 vex   chained    r"vxorps ymm\v, ymm\v, ymm\fixed"                 "_mm256_xor_pd _mm256_xor_ps _mm256_xor_si256";
    KSHIFTLQ         "k, k, imm8"            "u64"  "u64"  avx512  avx512bw            k     chained    r"kshiftlq k\v, k\v, 1"                           "_kshiftli_mask64";
    KSHIFTRQ         "k, k, imm8"            "u64"  "u64"  avx512  avx512bw            k     chained    r"kshiftrq k\v, k\v, 1"                           "_kshiftri_mask64";
    VADDPD           "zmm {k}, zmm, zmm"     "f64"  "f64"  avx512  avx512f             evex  chained    r"vaddpd zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_add_pd";
    VADDPD           "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vaddpd zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_pd";
    VADDPH           "ymm, ymm, ymm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vaddph ymm\v, ymm\v, ymm\fixed"                 "_mm256_add_ph";
    VADDPH           "zmm {k}, zmm, zmm"     "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vaddph zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_add_ph";
    VADDPH           "zmm, zmm, zmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vaddph zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_ph";
    VADDPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vaddps zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_ps";
    VALIGND          "zmm, zmm, zmm, imm8"   "u32"  "u32"  avx512  avx512f             evex  chained    r"valignd zmm\v, zmm\v, zmm\fixed, 1"             "_mm512_alignr_epi32";
    VALIGNQ          "zmm, zmm, zmm, imm8"   "u64"  "u64"  avx512  avx512f             evex  chained    r"valignq zmm\v, zmm\v, zmm\fixed, 1"             "_mm512_alignr_epi64";
    VANDNPS          "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512dq            evex  chained    r"vandnps zmm\v, zmm\v, zmm\fixed"                "_mm512_andnot_si512";
    VANDPD           "zmm, zmm, m64{1to8}"   "f64"  "f64"  avx512  avx512dq            lines throughput r"vandpd zmm0, zmm31, qword ptr [\v]{{1to8}}"     "_mm512_abs_pd _mm512_abs_ps _mm512_abs_ph";
    VANDPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512dq            evex  chained    r"vandps zmm\v, zmm\v, zmm\fixed"                 "_mm512_and_pd _mm512_and_ps _mm512_and_si512";
    VBROADCASTSD     "zmm, xmm"              "f64"  "f64"  avx512  avx512f             evex  chained    r"vbroadcastsd zmm\v, xmm\v"                      "_mm512_set1_pd";
    VBROADCASTSS     "zmm, xmm"              "f32"  "f32"  avx512  avx512f             evex  chained    r"vbroadcastss zmm\v, xmm\v"                      "_mm512_broadcastd_epi32 _mm512_broadcastss_ps _mm512_set1_ps";
    VCMPGTSH         "k, xmm, xmm"           "f16"  "f16"  avx512  avx512_fp16         evex  throughput r"vcmpgtsh k1, xmm\v, xmm\fixed"                  "_mm_comigt_sh";
    VCMPLTPD         "k, ymm, ymm"           "f64"  "f64"  avx512  avx512vl            evex  throughput r"vcmpltpd k1, ymm\v, ymm\fixed"                  "_mm256_cmp_pd_mask";
    VCMPLTPD         "k, zmm, zmm"           "f64"  "f64"  avx512  avx512f             evex  throughput r"vcmpltpd k1, zmm\v, zmm\fixed"                  "_mm512_cmp_pd_mask";
    VCMPLTPH         "k, ymm, ymm"           "f16"  "f16"  avx512  avx512_fp16         evex  throughput r"vcmpltph k1, ymm\v, ymm\fixed"                  "_mm256_cmp_ph_mask";
    VCMPLTPH         "k, zmm, zmm"           "f16"  "f16"  avx512  avx512_fp16         evex  throughput r"vcmpltph k1, zmm\v, zmm\fixed"                  "_mm512_cmp_ph_mask";
    VCMPLTPS         "k, xmm, xmm"           "f32"  "f32"  avx512  avx512vl            evex  throughput r"vcmpltps k1, xmm\v, xmm\fixed"                  "_mm_cmp_ps_mask";
    VCMPLTPS         "k, ymm, ymm"           "f32"  "f32"  avx512  avx512vl            evex  throughput r"vcmpltps k1, ymm\v, ymm\fixed"                  "_mm256_cmp_ps_mask";
    VCMPLTPS         "k, zmm, zmm"           "f32"  "f32"  avx512  avx512f             evex  throughput r"vcmpltps k1, zmm\v, zmm\fixed"                  "_mm512_cmp_ps_mask";
    VCMPLTSD         "k, xmm, xmm"           "f64"  "f64"  avx512  avx512vl            evex  throughput r"vcmpltsd k1, xmm\v, xmm\fixed"                  "_mm_cmp_sd_mask";
    VCMPLTSH         "k, xmm, xmm"           "f16"  "f16"  avx512  avx512_fp16         evex  throughput r"vcmpltsh k1, xmm\v, xmm\fixed"                  "_mm_comilt_sh";
    VCMPNEQPD        "k, zmm, zmm"           "f64"  "f64"  avx512  avx512f             evex  throughput r"vcmpneqpd k1, zmm\v, zmm\fixed"                 "_mm512_cmpneq_pd_mask";
    VCOMPRESSPS      "zmm {k} {z}, zmm"      "f32"  "f32"  avx512  avx512f             evex  chained    r"vcompressps zmm\v {{k1}} {{z}}, zmm\v"          "_mm512_maskz_compress_ps";
    VCVTDQ2PD        "zmm, ymm"              "i32"  "f64"  avx512  avx512f             evex  chained    r"vcvtdq2pd zmm\v, ymm\v"                         "_mm512_cvtepi32_pd";
    VCVTDQ2PS        "zmm, zmm"              "i32"  "f32"  avx512  avx512f             evex  chained    r"vcvtdq2ps zmm\v, zmm\v"                         "_mm512_cvtepi32_ps";
    VCVTNEPS2BF16    "ymm, zmm"              "f32"  "bf16" avx512  avx512_bf16         evex  chained    r"vcvtneps2bf16 ymm\v, zmm\v"                     "_mm512_cvtneps_pbh";
    VCVTPD2DQ        "ymm {k} {z}, zmm"      "f64"  "i32"  avx512  avx512f             evex  chained    r"vcvtpd2dq ymm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_cvtpd_epi32";
    VCVTPD2DQ        "ymm, zmm"              "f64"  "i32"  avx512  avx512f             evex  chained    r"vcvtpd2dq ymm\v, zmm\v"                         "_mm512_cvt_roundpd_epi32";
    VCVTPD2PS        "ymm, zmm"              "f64"  "f32"  avx512  avx512f             evex  chained    r"vcvtpd2ps ymm\v, zmm\v"                         "_mm512_cvtpd_ps";
    VCVTPD2QQ        "zmm {k} {z}, zmm"      "f64"  "i64"  avx512  avx512dq            evex  chained    r"vcvtpd2qq zmm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_cvtpd_epi64";
    VCVTPD2UDQ       "ymm, zmm"              "f64"  "u32"  avx512  avx512f             evex  chained    r"vcvtpd2udq ymm\v, zmm\v"                        "_mm512_cvtpd_epu32";
    VCVTPD2UQQ       "zmm, zmm"              "f64"  "u64"  avx512  avx512dq            evex  chained    r"vcvtpd2uqq zmm\v, zmm\v"                        "_mm512_cvtpd_epu64";
    VCVTPH2PSX       "xmm, xmm"              "f16"  "f32"  avx512  avx512_fp16         evex  chained    r"vcvtph2psx xmm\v, xmm\v"                        "_mm_cvtph_ps";
    VCVTPH2PSX       "ymm, xmm"              "f16"  "f32"  avx512  avx512_fp16         evex  chained    r"vcvtph2psx ymm\v, xmm\v"                        "_mm256_cvtph_ps";
    VCVTPH2PSX       "zmm {k}, ymm"          "f16"  "f32"  avx512  avx512_fp16         evex  chained    r"vcvtph2psx zmm\v {{k1}}, ymm\v"                 "_mm512_mask_cvtph_ps";
    VCVTPH2PSX       "zmm, ymm"              "f16"  "f32"  avx512  avx512_fp16         evex  chained    r"vcvtph2psx zmm\v, ymm\v"                        "_mm512_cvtph_ps _mm512_cvtxph_ps";
    VCVTPH2W         "ymm, ymm"              "f16"  "i16"  avx512  avx512_fp16         evex  chained    r"vcvtph2w ymm\v, ymm\v"                          "_mm256_cvtph_epi16";
    VCVTPS2DQ        "zmm {k} {z}, zmm"      "f32"  "i32"  avx512  avx512f             evex  chained    r"vcvtps2dq zmm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_cvtps_epi32";
    VCVTPS2DQ        "zmm, zmm"              "f32"  "i32"  avx512  avx512f             evex  chained    r"vcvtps2dq zmm\v, zmm\v"                         "_mm512_cvt_roundps_epi32 _mm512_cvtps_epi32";
    VCVTPS2PD        "zmm {k}, ymm"          "f32"  "f64"  avx512  avx512f             evex  chained    r"vcvtps2pd zmm\v {{k1}}, ymm\v"                  "_mm512_mask_cvtps_pd";
    VCVTPS2PD        "zmm, ymm"              "f32"  "f64"  avx512  avx512f             evex  chained    r"vcvtps2pd zmm\v, ymm\v"                         "_mm512_cvtps_pd";
    VCVTPS2PH        "ymm, zmm, imm8"        "f32"  "f16"  avx512  avx512f             evex  chained    r"vcvtps2ph ymm\v, zmm\v, 0"                      "_mm512_cvtps_ph";
    VCVTPS2PHX       "ymm, zmm"              "f32"  "f16"  avx512  avx512_fp16         evex  chained    r"vcvtps2phx ymm\v, zmm\v"                        "_mm512_cvtxps_ph";
    VCVTPS2UDQ       "zmm, zmm"              "f32"  "u32"  avx512  avx512f             evex  chained    r"vcvtps2udq zmm\v, zmm\v"                        "_mm512_cvtps_epu32";
    VCVTQQ2PD        "zmm, zmm"              "i64"  "f64"  avx512  avx512dq            evex  chained    r"vcvtqq2pd zmm\v, zmm\v"                         "_mm512_cvtepi64_pd";
    VCVTQQ2PS        "ymm, zmm"              "i64"  "f32"  avx512  avx512dq            evex  chained    r"vcvtqq2ps ymm\v, zmm\v"                         "_mm512_cvtepi64_ps";
    VCVTSH2SS        "xmm, xmm, xmm"         "f16"  "f32"  avx512  avx512_fp16         evex  chained    r"vcvtsh2ss xmm\v, xmm\fixed, xmm\v"              "_mm_cvtsh_ss";
    VCVTSS2SH        "xmm, xmm, xmm"         "f32"  "f16"  avx512  avx512_fp16         evex  chained    r"vcvtss2sh xmm\v, xmm\fixed, xmm\v"              "_mm_cvtss_sh";
    VCVTTPD2UDQ      "ymm, zmm"              "f64"  "u32"  avx512  avx512f             evex  chained    r"vcvttpd2udq ymm\v, zmm\v"                       "_mm512_cvttpd_epu32";
    VCVTTPH2W        "zmm, zmm"              "f16"  "i16"  avx512  avx512_fp16         evex  chained    r"vcvttph2w zmm\v, zmm\v"                         "_mm512_cvttph_epi16";
    VCVTTPS2DQ       "zmm, zmm"              "f32"  "i32"  avx512  avx512f             evex  chained    r"vcvttps2dq zmm\v, zmm\v"                        "_mm512_cvttps_epi32";
    VCVTUDQ2PD       "zmm, ymm"              "u32"  "f64"  avx512  avx512f             evex  chained    r"vcvtudq2pd zmm\v, ymm\v"                        "_mm512_cvtepu32_pd";
    VCVTUDQ2PS       "zmm, zmm"              "u32"  "f32"  avx512  avx512f             evex  chained    r"vcvtudq2ps zmm\v, zmm\v"                        "_mm512_cvtepu32_ps";
    VCVTUQQ2PD       "zmm, zmm"              "u64"  "f64"  avx512  avx512dq            evex  chained    r"vcvtuqq2pd zmm\v, zmm\v"                        "_mm512_cvtepu64_pd";
    VCVTUQQ2PS       "ymm, zmm"              "u64"  "f32"  avx512  avx512dq            evex  chained    r"vcvtuqq2ps ymm\v, zmm\v"                        "_mm512_cvtepu64_ps";
    VCVTW2PH         "ymm, ymm"              "i16"  "f16"  avx512  avx512_fp16         evex  chained    r"vcvtw2ph ymm\v, ymm\v"                          "_mm256_cvtepi16_ph";
    VDIVPD           "zmm {k} {z}, zmm, zmm" "f64"  "f64"  avx512  avx512f             evex  chained    r"vdivpd zmm\v {{k1}} {{z}}, zmm\fixed, zmm\v"    "_mm512_maskz_div_pd";
    VDIVPD           "zmm {k}, zmm, zmm"     "f64"  "f64"  avx512  avx512f             evex  chained    r"vdivpd zmm\v {{k1}}, zmm\fixed, zmm\v"          "_mm512_mask_div_pd";
    VDIVPD           "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vdivpd zmm\v, zmm\fixed, zmm\v"                 "_mm512_div_pd";
    VDIVPH           "zmm {k}, zmm, zmm"     "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vdivph zmm\v {{k1}}, zmm\fixed, zmm\v"          "_mm512_mask_div_ph";
    VDIVPS           "zmm {k}, zmm, zmm"     "f32"  "f32"  avx512  avx512f             evex  chained    r"vdivps zmm\v {{k1}}, zmm\fixed, zmm\v"          "_mm512_mask_div_ps";
    VDIVPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vdivps zmm\v, zmm\fixed, zmm\v"                 "_mm512_div_ps";
    VDPBF16PS        "ymm, ymm, ymm"         "bf16" "f32"  avx512  avx512_bf16         evex  chained    r"vdpbf16ps ymm\v, ymm\fixed, ymm\fixed"          "_mm256_dpbf16_ps";
    VDPBF16PS        "zmm, zmm, zmm"         "bf16" "f32"  avx512  avx512_bf16         evex  chained    r"vdpbf16ps zmm\v, zmm\fixed, zmm\fixed"          "_mm512_dpbf16_ps";
    VEXTRACTF64X4    "ymm, zmm, imm8"        "f64"  "f64"  avx512  avx512f             evex  chained    r"vextractf64x4 ymm\v, zmm\v, 1"                  "_mm512_extractf32x8_ps _mm512_extracti32x8_epi32 _mm512_extractf64x4_pd _mm512_extracti64x4_epi64";
    VFMADD132PS      "zmm {k}, zmm, zmm"     "f32"  "f32"  avx512  avx512f             evex  chained    r"vfmadd132ps zmm\v {{k1}}, zmm\fixed, zmm\fixed" "_mm512_mask_fmadd_ps";
    VFMADD213PD      "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vfmadd213pd zmm\v, zmm\fixed, zmm\fixed"        "_mm512_fmadd_pd";
    VFMADD213PH      "zmm, zmm, zmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vfmadd213ph zmm\v, zmm\fixed, zmm\fixed"        "_mm512_fmadd_ph";
    VFMADD213PS      "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vfmadd213ps zmm\v, zmm\fixed, zmm\fixed"        "_mm512_fmadd_ps";
    VFMADD213SH      "xmm, xmm, xmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vfmadd213sh xmm\v, xmm\fixed, xmm\fixed"        "_mm_fmadd_sh";
    VFMADDSUB213PD   "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vfmaddsub213pd zmm\v, zmm\fixed, zmm\fixed"     "_mm512_fmaddsub_pd";
    VFMADDSUB213PS   "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vfmaddsub213ps zmm\v, zmm\fixed, zmm\fixed"     "_mm512_fmaddsub_ps";
    VFMSUB213PD      "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vfmsub213pd zmm\v, zmm\fixed, zmm\fixed"        "_mm512_fmsub_pd";
    VFMSUB213PS      "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vfmsub213ps zmm\v, zmm\fixed, zmm\fixed"        "_mm512_fmsub_ps";
    VFNMADD213PD     "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vfnmadd213pd zmm\v, zmm\fixed, zmm\fixed"       "_mm512_fnmadd_pd";
    VFNMADD213PH     "zmm, zmm, zmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vfnmadd213ph zmm\v, zmm\fixed, zmm\fixed"       "_mm512_fnmadd_ph";
    VFNMADD213PS     "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vfnmadd213ps zmm\v, zmm\fixed, zmm\fixed"       "_mm512_fnmadd_ps";
    VFPCLASSPS       "k, zmm, imm8"          "f32"  "f32"  avx512  avx512dq            evex  throughput r"vfpclassps k1, zmm\v, 1"                        "_mm512_fpclass_ps_mask";
    VGETEXPPD        "zmm, zmm"              "f64"  "f64"  avx512  avx512f             evex  chained    r"vgetexppd zmm\v, zmm\v"                         "_mm512_getexp_pd";
    VGETEXPPS        "xmm, xmm"              "f32"  "f32"  avx512  avx512vl            evex  chained    r"vgetexpps xmm\v, xmm\v"                         "_mm_getexp_ps";
    VGETEXPPS        "zmm, zmm"              "f32"  "f32"  avx512  avx512f             evex  chained    r"vgetexpps zmm\v, zmm\v"                         "_mm512_getexp_ps";
    VGETMANTPD       "zmm, zmm, imm8"        "f64"  "f64"  avx512  avx512f             evex  chained    r"vgetmantpd zmm\v, zmm\v, 0"                     "_mm512_getmant_pd";
    VGETMANTPS       "zmm, zmm, imm8"        "f32"  "f32"  avx512  avx512f             evex  chained    r"vgetmantps zmm\v, zmm\v, 0"                     "_mm512_getmant_ps";
    VINSERTF32X4     "zmm, zmm, xmm, imm8"   "f32"  "f32"  avx512  avx512f             evex  chained    r"vinsertf32x4 zmm\v, zmm\v, xmm\fixed, 1"        "_mm512_inserti64x2 _mm512_inserti32x4";
    VINSERTF64X4     "zmm, zmm, ymm, imm8"   "f64"  "f64"  avx512  avx512f             evex  chained    r"vinsertf64x4 zmm\v, zmm\v, ymm\fixed, 1"        "_mm512_insertf32x8 _mm512_broadcast_i64x4 _mm512_inserti64x4";
    VMAXPD           "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vmaxpd zmm\v, zmm\v, zmm\fixed"                 "_mm512_max_pd";
    VMAXPS           "zmm {k}, zmm, zmm"     "f32"  "f32"  avx512  avx512f             evex  chained    r"vmaxps zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_max_ps";
    VMAXPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vmaxps zmm\v, zmm\v, zmm\fixed"                 "_mm512_max_ps";
    VMINPD           "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vminpd zmm\v, zmm\v, zmm\fixed"                 "_mm512_min_pd";
    VMINPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vminps zmm\v, zmm\v, zmm\fixed"                 "_mm512_min_ps";
    VMOVAPD          "ymm {k}, ymm"          "f64"  "f64"  avx512  avx512vl            evex  chained    r"vmovapd ymm\v {{k1}}, ymm\v"                    "_mm256_mask_mov_pd";
    VMOVAPD          "zmm {k} {z}, zmm"      "f64"  "f64"  avx512  avx512f             evex  chained    r"vmovapd zmm\v {{k1}} {{z}}, zmm\v"              "_mm512_maskz_mov_pd";
    VMOVAPD          "zmm {k}, zmm"          "f64"  "f64"  avx512  avx512f             evex  chained    r"vmovapd zmm\v {{k1}}, zmm\v"                    "_mm512_mask_blend_pd _mm512_mask_mov_pd";
    VMOVAPS          "m512, zmm"             "f32"  "f32"  avx512  avx512f             lines throughput r"vmovaps zmmword ptr [\v], zmm0"                 "_mm512_store_ps _mm512_store_si512";
    VMOVAPS          "xmm {k}, xmm"          "f32"  "f32"  avx512  avx512vl            evex  chained    r"vmovaps xmm\v {{k1}}, xmm\v"                    "_mm_mask_mov_ps";
    VMOVAPS          "zmm {k} {z}, zmm"      "f32"  "f32"  avx512  avx512f             evex  chained    r"vmovaps zmm\v {{k1}} {{z}}, zmm\v"              "_mm512_maskz_mov_ps";
    VMOVAPS          "zmm {k}, zmm"          "f32"  "f32"  avx512  avx512f             evex  chained    r"vmovaps zmm\v {{k1}}, zmm\v"                    "_mm512_mask_blend_ps _mm512_mask_mov_ps";
    VMOVAPS          "zmm, m512"             "f32"  "f32"  avx512  avx512f             lines throughput r"vmovaps zmm0, zmmword ptr [\v]"                 "_mm512_load_epi64 _mm512_load_ps _mm512_load_si512";
    VMOVDQA32        "zmm {k} {z}, zmm"      "u32"  "u32"  avx512  avx512f             evex  chained    r"vmovdqa32 zmm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_mov_epi32";
    VMOVDQA32        "zmm {k}, zmm"          "u32"  "u32"  avx512  avx512f             evex  chained    r"vmovdqa32 zmm\v {{k1}}, zmm\v"                  "_mm512_mask_blend_epi32 _mm512_mask_mov_epi32";
    VMOVDQA64        "zmm {k} {z}, zmm"      "u64"  "u64"  avx512  avx512f             evex  chained    r"vmovdqa64 zmm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_mov_epi64";
    VMOVDQA64        "zmm {k}, zmm"          "u64"  "u64"  avx512  avx512f             evex  chained    r"vmovdqa64 zmm\v {{k1}}, zmm\v"                  "_mm512_mask_blend_epi64 _mm512_mask_mov_epi64";
    VMOVDQU16        "m128 {k}, xmm"         "u16"  "u16"  avx512  avx512bw            lines throughput r"vmovdqu16 xmmword ptr [\v] {{k1}}, xmm0"        "_mm_mask_storeu_epi16";
    VMOVDQU16        "m256 {k}, ymm"         "u16"  "u16"  avx512  avx512bw            lines throughput r"vmovdqu16 ymmword ptr [\v] {{k1}}, ymm0"        "_mm256_mask_storeu_epi16";
    VMOVDQU16        "m512 {k}, zmm"         "u16"  "u16"  avx512  avx512bw            lines throughput r"vmovdqu16 zmmword ptr [\v] {{k1}}, zmm0"        "_mm512_mask_storeu_epi16";
    VMOVDQU16        "xmm {k} {z}, m128"     "u16"  "u16"  avx512  avx512bw            lines throughput r"vmovdqu16 xmm0 {{k1}} {{z}}, xmmword ptr [\v]"  "_mm_maskz_loadu_epi16";
    VMOVDQU16        "ymm {k} {z}, m256"     "u16"  "u16"  avx512  avx512bw            lines throughput r"vmovdqu16 ymm0 {{k1}} {{z}}, ymmword ptr [\v]"  "_mm256_maskz_loadu_epi16";
    VMOVDQU16        "ymm {k}, ymm"          "u16"  "u16"  avx512  avx512bw            evex  chained    r"vmovdqu16 ymm\v {{k1}}, ymm\v"                  "_mm256_mask_blend_epi16 _mm256_mask_mov_epi16";
    VMOVDQU16        "zmm {k} {z}, m512"     "u16"  "u16"  avx512  avx512bw            lines throughput r"vmovdqu16 zmm0 {{k1}} {{z}}, zmmword ptr [\v]"  "_mm512_maskz_loadu_epi16";
    VMOVDQU16        "zmm {k} {z}, zmm"      "u16"  "u16"  avx512  avx512bw            evex  chained    r"vmovdqu16 zmm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_mov_epi16";
    VMOVDQU16        "zmm {k}, zmm"          "u16"  "u16"  avx512  avx512bw            evex  chained    r"vmovdqu16 zmm\v {{k1}}, zmm\v"                  "_mm512_mask_blend_epi16 _mm512_mask_mov_epi16";
    VMOVDQU32        "m128 {k}, xmm"         "u32"  "u32"  avx512  avx512vl            lines throughput r"vmovdqu32 xmmword ptr [\v] {{k1}}, xmm0"        "_mm_mask_storeu_epi32";
    VMOVDQU32        "m256 {k}, ymm"         "u32"  "u32"  avx512  avx512vl            lines throughput r"vmovdqu32 ymmword ptr [\v] {{k1}}, ymm0"        "_mm256_mask_storeu_epi32";
    VMOVDQU32        "m512 {k}, zmm"         "u32"  "u32"  avx512  avx512f             lines throughput r"vmovdqu32 zmmword ptr [\v] {{k1}}, zmm0"        "_mm512_mask_storeu_epi32";
    VMOVDQU32        "xmm {k} {z}, m128"     "u32"  "u32"  avx512  avx512vl            lines throughput r"vmovdqu32 xmm0 {{k1}} {{z}}, xmmword ptr [\v]"  "_mm_maskz_loadu_epi32";
    VMOVDQU32        "ymm {k} {z}, m256"     "u32"  "u32"  avx512  avx512vl            lines throughput r"vmovdqu32 ymm0 {{k1}} {{z}}, ymmword ptr [\v]"  "_mm256_maskz_loadu_epi32";
    VMOVDQU32        "zmm {k} {z}, m512"     "u32"  "u32"  avx512  avx512f             lines throughput r"vmovdqu32 zmm0 {{k1}} {{z}}, zmmword ptr [\v]"  "_mm512_maskz_loadu_epi32";
    VMOVDQU64        "m256 {k}, ymm"         "u64"  "u64"  avx512  avx512vl            lines throughput r"vmovdqu64 ymmword ptr [\v] {{k1}}, ymm0"        "_mm256_mask_storeu_epi64";
    VMOVDQU64        "m512 {k}, zmm"         "u64"  "u64"  avx512  avx512f             lines throughput r"vmovdqu64 zmmword ptr [\v] {{k1}}, zmm0"        "_mm512_mask_storeu_epi64";
    VMOVDQU64        "ymm {k} {z}, m256"     "u64"  "u64"  avx512  avx512vl            lines throughput r"vmovdqu64 ymm0 {{k1}} {{z}}, ymmword ptr [\v]"  "_mm256_maskz_loadu_epi64";
    VMOVDQU64        "zmm {k} {z}, m512"     "u64"  "u64"  avx512  avx512f             lines throughput r"vmovdqu64 zmm0 {{k1}} {{z}}, zmmword ptr [\v]"  "_mm512_maskz_loadu_epi64";
    VMOVDQU8         "m128 {k}, xmm"         "u8"   "u8"   avx512  avx512bw            lines throughput r"vmovdqu8 xmmword ptr [\v] {{k1}}, xmm0"         "_mm_mask_storeu_epi8";
    VMOVDQU8         "m256 {k}, ymm"         "u8"   "u8"   avx512  avx512bw            lines throughput r"vmovdqu8 ymmword ptr [\v] {{k1}}, ymm0"         "_mm256_mask_storeu_epi8";
    VMOVDQU8         "m512 {k}, zmm"         "u8"   "u8"   avx512  avx512bw            lines throughput r"vmovdqu8 zmmword ptr [\v] {{k1}}, zmm0"         "_mm512_mask_storeu_epi8";
    VMOVDQU8         "xmm {k} {z}, m128"     "u8"   "u8"   avx512  avx512bw            lines throughput r"vmovdqu8 xmm0 {{k1}} {{z}}, xmmword ptr [\v]"   "_mm_maskz_loadu_epi8";
    VMOVDQU8         "xmm {k}, xmm"          "u8"   "u8"   avx512  avx512bw            evex  chained    r"vmovdqu8 xmm\v {{k1}}, xmm\v"                   "_mm_mask_blend_epi8";
    VMOVDQU8         "ymm {k} {z}, m256"     "u8"   "u8"   avx512  avx512bw            lines throughput r"vmovdqu8 ymm0 {{k1}} {{z}}, ymmword ptr [\v]"   "_mm256_maskz_loadu_epi8";
    VMOVDQU8         "ymm {k}, ymm"          "u8"   "u8"   avx512  avx512bw            evex  chained    r"vmovdqu8 ymm\v {{k1}}, ymm\v"                   "_mm256_mask_mov_epi8";
    VMOVDQU8         "zmm {k} {z}, m512"     "u8"   "u8"   avx512  avx512bw            lines throughput r"vmovdqu8 zmm0 {{k1}} {{z}}, zmmword ptr [\v]"   "_mm512_maskz_loadu_epi8";
    VMOVDQU8         "zmm {k} {z}, zmm"      "u8"   "u8"   avx512  avx512bw            evex  chained    r"vmovdqu8 zmm\v {{k1}} {{z}}, zmm\v"             "_mm512_maskz_mov_epi8";
    VMOVDQU8         "zmm {k}, zmm"          "u8"   "u8"   avx512  avx512bw            evex  chained    r"vmovdqu8 zmm\v {{k1}}, zmm\v"                   "_mm512_mask_blend_epi8 _mm512_mask_mov_epi8";
    VMOVNTDQA        "zmm, m512"             "u8"   "u8"   avx512  avx512f             lines throughput r"vmovntdqa zmm0, zmmword ptr [\v]"               "_mm512_stream_load_si512";
    VMOVSD           "xmm {k}, xmm, xmm"     "f64"  "f64"  avx512  avx512vl            evex  chained    r"vmovsd xmm\v {{k1}}, xmm\v, xmm\fixed"          "_mm_mask_move_sd";
    VMOVSH           "xmm, xmm, xmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vmovsh xmm\v, xmm\v, xmm\fixed"                 "_mm_blend_epi16";
    VMOVSHDUP        "zmm, zmm"              "f32"  "f32"  avx512  avx512f             evex  chained    r"vmovshdup zmm\v, zmm\v"                         "_mm512_movehdup_ps";
    VMOVSLDUP        "zmm, zmm"              "f32"  "f32"  avx512  avx512f             evex  chained    r"vmovsldup zmm\v, zmm\v"                         "_mm512_moveldup_ps";
    VMOVUPD          "m512 {k}, zmm"         "f64"  "f64"  avx512  avx512f             lines throughput r"vmovupd zmmword ptr [\v] {{k1}}, zmm0"          "_mm512_mask_storeu_pd";
    VMOVUPD          "zmm {k} {z}, m512"     "f64"  "f64"  avx512  avx512f             lines throughput r"vmovupd zmm0 {{k1}} {{z}}, zmmword ptr [\v]"    "_mm512_maskz_loadu_pd";
    VMOVUPS          "m128 {k}, xmm"         "f32"  "f32"  avx512  avx512vl            lines throughput r"vmovups xmmword ptr [\v] {{k1}}, xmm0"          "_mm_mask_storeu_ps";
    VMOVUPS          "m256 {k}, ymm"         "f32"  "f32"  avx512  avx512vl            lines throughput r"vmovups ymmword ptr [\v] {{k1}}, ymm0"          "_mm256_mask_storeu_ps";
    VMOVUPS          "m512 {k}, zmm"         "f32"  "f32"  avx512  avx512f             lines throughput r"vmovups zmmword ptr [\v] {{k1}}, zmm0"          "_mm512_mask_storeu_ps";
    VMOVUPS          "m512, zmm"             "f32"  "f32"  avx512  avx512f             lines throughput r"vmovups zmmword ptr [\v], zmm0"                 "_mm512_storeu_epi8 _mm512_storeu_pd _mm512_storeu_ps _mm512_storeu_si512";
    VMOVUPS          "xmm {k} {z}, m128"     "f32"  "f32"  avx512  avx512vl            lines throughput r"vmovups xmm0 {{k1}} {{z}}, xmmword ptr [\v]"    "_mm_maskz_loadu_ps";
    VMOVUPS          "ymm {k} {z}, m256"     "f32"  "f32"  avx512  avx512vl            lines throughput r"vmovups ymm0 {{k1}} {{z}}, ymmword ptr [\v]"    "_mm256_maskz_loadu_ps";
    VMOVUPS          "zmm {k} {z}, m512"     "f32"  "f32"  avx512  avx512f             lines throughput r"vmovups zmm0 {{k1}} {{z}}, zmmword ptr [\v]"    "_mm512_maskz_loadu_ps";
    VMOVUPS          "zmm, m512"             "f32"  "f32"  avx512  avx512f             lines throughput r"vmovups zmm0, zmmword ptr [\v]"                 "_mm512_loadu_epi16 _mm512_loadu_epi8 _mm512_loadu_epi32 _mm512_loadu_epi64 _mm512_loadu_pd _mm512_loadu_ps _mm512_loadu_si512";
    VMULPD           "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vmulpd zmm\v, zmm\v, zmm\v"                     "_mm512_mul_pd";
    VMULPH           "ymm, ymm, ymm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vmulph ymm\v, ymm\v, ymm\v"                     "_mm256_mul_ph";
    VMULPH           "zmm, zmm, zmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vmulph zmm\v, zmm\v, zmm\v"                     "_mm512_mul_ph";
    VMULPS           "zmm {k} {z}, zmm, zmm" "f32"  "f32"  avx512  avx512f             evex  chained    r"vmulps zmm\v {{k1}} {{z}}, zmm\v, zmm\v"        "_mm512_maskz_mul_ps";
    VMULPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vmulps zmm\v, zmm\v, zmm\v"                     "_mm512_mul_ps";
    VORPS            "zmm {k}, zmm, zmm"     "f32"  "f32"  avx512  avx512dq            evex  chained    r"vorps zmm\v {{k1}}, zmm\v, zmm\fixed"           "_mm512_mask_or_ps _mm512_mask_or_epi32";
    VORPS            "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512dq            evex  chained    r"vorps zmm\v, zmm\v, zmm\fixed"                  "_mm512_or_ps _mm512_or_si512";
    VP2INTERSECTD    "k, zmm, zmm"           "i32"  "i32"  avx512  avx512_vp2intersect evex  throughput r"vp2intersectd k2, zmm\v, zmm\fixed"             "_mm512_2intersect_epi32";
    VP2INTERSECTQ    "k, zmm, zmm"           "i64"  "i64"  avx512  avx512_vp2intersect evex  throughput r"vp2intersectq k2, zmm\v, zmm\fixed"             "_mm512_2intersect_epi64";
    VPABSB           "zmm, zmm"              "i8"   "i8"   avx512  avx512bw            evex  chained    r"vpabsb zmm\v, zmm\v"                            "_mm512_abs_epi8";
    VPABSQ           "zmm, zmm"              "i64"  "i64"  avx512  avx512f             evex  chained    r"vpabsq zmm\v, zmm\v"                            "_mm512_abs_epi64";
    VPACKUSDW        "zmm, zmm, zmm"         "i32"  "u16"  avx512  avx512bw            evex  chained    r"vpackusdw zmm\v, zmm\v, zmm\fixed"              "_mm512_packus_epi32";
    VPACKUSWB        "zmm, zmm, zmm"         "i16"  "u8"   avx512  avx512bw            evex  chained    r"vpackuswb zmm\v, zmm\v, zmm\fixed"              "_mm512_packus_epi16";
    VPADDB           "zmm {k}, zmm, zmm"     "i8"   "i8"   avx512  avx512bw            evex  chained    r"vpaddb zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_add_epi8";
    VPADDB           "zmm, zmm, zmm"         "i8"   "i8"   avx512  avx512bw            evex  chained    r"vpaddb zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_epi8";
    VPADDD           "zmm {k}, zmm, zmm"     "i32"  "i32"  avx512  avx512f             evex  chained    r"vpaddd zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_add_epi32";
    VPADDD           "zmm, zmm, zmm"         "i32"  "i32"  avx512  avx512f             evex  chained    r"vpaddd zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_epi32";
    VPADDQ           "zmm {k}, zmm, zmm"     "i64"  "i64"  avx512  avx512f             evex  chained    r"vpaddq zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_add_epi64";
    VPADDQ           "zmm, zmm, zmm"         "i64"  "i64"  avx512  avx512f             evex  chained    r"vpaddq zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_epi64";
    VPADDSB          "zmm, zmm, zmm"         "i8"   "i8"   avx512  avx512bw            evex  chained    r"vpaddsb zmm\v, zmm\v, zmm\fixed"                "_mm512_adds_epi8";
    VPADDSW          "zmm, zmm, zmm"         "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpaddsw zmm\v, zmm\v, zmm\fixed"                "_mm512_adds_epi16";
    VPADDUSB         "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpaddusb zmm\v, zmm\v, zmm\fixed"               "_mm512_adds_epu8";
    VPADDUSW         "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpaddusw zmm\v, zmm\v, zmm\fixed"               "_mm512_adds_epu16";
    VPADDW           "zmm {k} {z}, zmm, zmm" "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpaddw zmm\v {{k1}} {{z}}, zmm\v, zmm\fixed"    "_mm512_maskz_add_epi16";
    VPADDW           "zmm {k}, zmm, zmm"     "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpaddw zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_add_epi16";
    VPADDW           "zmm, zmm, zmm"         "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpaddw zmm\v, zmm\v, zmm\fixed"                 "_mm512_add_epi16";
    VPBROADCASTB     "xmm, r32"              "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpbroadcastb xmm\v, eax"                        "_mm_set1_epi8";
    VPBROADCASTB     "ymm, r32"              "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpbroadcastb ymm\v, eax"                        "_mm256_set1_epi8";
    VPBROADCASTB     "zmm {k}, r32"          "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpbroadcastb zmm\v {{k1}}, eax"                 "_mm512_mask_set1_epi8";
    VPBROADCASTB     "zmm, r32"              "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpbroadcastb zmm\v, eax"                        "_mm512_set1_epi8";
    VPBROADCASTD     "xmm, r32"              "u32"  "u32"  avx512  avx512vl            evex  throughput r"vpbroadcastd xmm\v, eax"                        "_mm_set1_epi32";
    VPBROADCASTD     "ymm, r32"              "u32"  "u32"  avx512  avx512vl            evex  throughput r"vpbroadcastd ymm\v, eax"                        "_mm256_set1_epi32";
    VPBROADCASTD     "zmm {k} {z}, r32"      "u32"  "u32"  avx512  avx512f             evex  throughput r"vpbroadcastd zmm\v {{k1}} {{z}}, eax"           "_mm512_maskz_set1_epi32";
    VPBROADCASTD     "zmm, r32"              "u32"  "u32"  avx512  avx512f             evex  throughput r"vpbroadcastd zmm\v, eax"                        "_mm512_set1_epi32";
    VPBROADCASTQ     "xmm, r64"              "u64"  "u64"  avx512  avx512vl            evex  throughput r"vpbroadcastq xmm\v, rax"                        "_mm_set1_epi64x";
    VPBROADCASTQ     "ymm, r64"              "u64"  "u64"  avx512  avx512vl            evex  throughput r"vpbroadcastq ymm\v, rax"                        "_mm256_set1_epi64x";
    VPBROADCASTQ     "zmm {k}, r64"          "u64"  "u64"  avx512  avx512f             evex  throughput r"vpbroadcastq zmm\v {{k1}}, rax"                 "_mm512_mask_set1_epi64";
    VPBROADCASTQ     "zmm, r64"              "u64"  "u64"  avx512  avx512f             evex  throughput r"vpbroadcastq zmm\v, rax"                        "_mm512_set1_epi64";
    VPBROADCASTW     "xmm, r32"              "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpbroadcastw xmm\v, eax"                        "_mm_set1_epi16";
    VPBROADCASTW     "ymm {k} {z}, r32"      "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpbroadcastw ymm\v {{k1}} {{z}}, eax"           "_mm256_maskz_set1_epi16";
    VPBROADCASTW     "ymm, r32"              "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpbroadcastw ymm\v, eax"                        "_mm256_set1_epi16";
    VPBROADCASTW     "zmm {k} {z}, r32"      "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpbroadcastw zmm\v {{k1}} {{z}}, eax"           "_mm512_maskz_set1_epi16";
    VPBROADCASTW     "zmm {k}, r32"          "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpbroadcastw zmm\v {{k1}}, eax"                 "_mm512_mask_set1_epi16";
    VPBROADCASTW     "zmm, r32"              "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpbroadcastw zmm\v, eax"                        "_mm512_set1_epi16";
    VPCMPEQB         "k, xmm, xmm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpeqb k1, xmm\v, xmm\fixed"                  "_mm_cmpeq_epi8_mask";
    VPCMPEQB         "k, ymm, ymm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpeqb k1, ymm\v, ymm\fixed"                  "_mm256_cmpeq_epi8_mask";
    VPCMPEQB         "k, zmm, zmm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpeqb k1, zmm\v, zmm\fixed"                  "_mm512_cmpeq_epi8_mask _mm512_cmpeq_epu8_mask";
    VPCMPEQD         "k, zmm, zmm"           "i32"  "i32"  avx512  avx512f             evex  throughput r"vpcmpeqd k1, zmm\v, zmm\fixed"                  "_mm512_cmpeq_epi32_mask";
    VPCMPEQQ         "k, zmm, zmm"           "i64"  "i64"  avx512  avx512f             evex  throughput r"vpcmpeqq k1, zmm\v, zmm\fixed"                  "_mm512_cmpeq_epi64_mask";
    VPCMPEQW         "k, ymm, ymm"           "i16"  "i16"  avx512  avx512bw            evex  throughput r"vpcmpeqw k1, ymm\v, ymm\fixed"                  "_mm256_cmpeq_epi16_mask";
    VPCMPEQW         "k, zmm, zmm"           "i16"  "i16"  avx512  avx512bw            evex  throughput r"vpcmpeqw k1, zmm\v, zmm\fixed"                  "_mm512_cmpeq_epi16_mask";
    VPCMPGTB         "k, zmm, zmm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpgtb k1, zmm\v, zmm\fixed"                  "_mm512_cmp_epi8_mask _mm512_cmplt_epi8_mask";
    VPCMPGTD         "k, zmm, zmm"           "i32"  "i32"  avx512  avx512f             evex  throughput r"vpcmpgtd k1, zmm\v, zmm\fixed"                  "_mm512_cmp_epi32_mask _mm512_cmpgt_epi32_mask";
    VPCMPGTQ         "k, zmm, zmm"           "i64"  "i64"  avx512  avx512f             evex  throughput r"vpcmpgtq k1, zmm\v, zmm\fixed"                  "_mm512_cmp_epi64_mask _mm512_cmpgt_epi64_mask";
    VPCMPGTW         "k, ymm, ymm"           "i16"  "i16"  avx512  avx512bw            evex  throughput r"vpcmpgtw k1, ymm\v, ymm\fixed"                  "_mm256_cmpgt_epi16_mask";
    VPCMPGTW         "k, zmm, zmm"           "i16"  "i16"  avx512  avx512bw            evex  throughput r"vpcmpgtw k1, zmm\v, zmm\fixed"                  "_mm512_cmp_epi16_mask _mm512_cmpgt_epi16_mask";
    VPCMPLEQ         "k, zmm, zmm"           "i64"  "i64"  avx512  avx512f             evex  throughput r"vpcmpleq k1, zmm\v, zmm\fixed"                  "_mm512_cmple_epi64_mask";
    VPCMPLEUB        "k, zmm, zmm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpcmpleub k1, zmm\v, zmm\fixed"                 "_mm512_cmple_epu8_mask";
    VPCMPLEUD        "k, zmm, zmm"           "u32"  "u32"  avx512  avx512f             evex  throughput r"vpcmpleud k1, zmm\v, zmm\fixed"                 "_mm512_cmple_epu32_mask";
    VPCMPLEUQ        "k, zmm, zmm"           "u64"  "u64"  avx512  avx512f             evex  throughput r"vpcmpleuq k1, zmm\v, zmm\fixed"                 "_mm512_cmple_epu64_mask";
    VPCMPLEUW        "k, ymm, ymm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpcmpleuw k1, ymm\v, ymm\fixed"                 "_mm256_cmple_epu16_mask";
    VPCMPLEUW        "k, zmm, zmm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpcmpleuw k1, zmm\v, zmm\fixed"                 "_mm512_cmple_epu16_mask";
    VPCMPLTUB        "k, zmm, zmm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpcmpltub k1, zmm\v, zmm\fixed"                 "_mm512_cmp_epu8_mask _mm512_cmplt_epu8_mask";
    VPCMPLTUD        "k, zmm, zmm"           "u32"  "u32"  avx512  avx512f             evex  throughput r"vpcmpltud k1, zmm\v, zmm\fixed"                 "_mm512_cmp_epu32_mask _mm512_cmplt_epu32_mask";
    VPCMPLTUQ        "k, zmm, zmm"           "u64"  "u64"  avx512  avx512f             evex  throughput r"vpcmpltuq k1, zmm\v, zmm\fixed"                 "_mm512_cmp_epu64_mask _mm512_cmplt_epu64_mask";
    VPCMPLTUW        "k, zmm, zmm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpcmpltuw k1, zmm\v, zmm\fixed"                 "_mm512_cmp_epu16_mask";
    VPCMPNEQB        "k, xmm, xmm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpneqb k1, xmm\v, xmm\fixed"                 "_mm_cmpneq_epi8_mask";
    VPCMPNEQB        "k, zmm, zmm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpneqb k1, zmm\v, zmm\fixed"                 "_mm512_cmpneq_epi8_mask";
    VPCMPNEQD        "k, zmm, zmm"           "i32"  "i32"  avx512  avx512f             evex  throughput r"vpcmpneqd k1, zmm\v, zmm\fixed"                 "_mm512_cmpneq_epi32_mask";
    VPCMPNEQQ        "k, zmm, zmm"           "i64"  "i64"  avx512  avx512f             evex  throughput r"vpcmpneqq k1, zmm\v, zmm\fixed"                 "_mm512_cmpneq_epi64_mask";
    VPCMPNEQW        "k, zmm, zmm"           "i16"  "i16"  avx512  avx512bw            evex  throughput r"vpcmpneqw k1, zmm\v, zmm\fixed"                 "_mm512_cmpneq_epi16_mask";
    VPCMPNLEUB       "k, ymm, ymm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpcmpnleub k1, ymm\v, ymm\fixed"                "_mm256_cmpgt_epu8_mask";
    VPCMPNLEUB       "k, zmm, zmm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpcmpnleub k1, zmm\v, zmm\fixed"                "_mm512_cmpgt_epu8_mask";
    VPCMPNLEUD       "k, zmm, zmm"           "u32"  "u32"  avx512  avx512f             evex  throughput r"vpcmpnleud k1, zmm\v, zmm\fixed"                "_mm512_cmpgt_epu32_mask";
    VPCMPNLEUQ       "k, zmm, zmm"           "u64"  "u64"  avx512  avx512f             evex  throughput r"vpcmpnleuq k1, zmm\v, zmm\fixed"                "_mm512_cmpgt_epu64_mask";
    VPCMPNLEUW       "k, ymm, ymm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpcmpnleuw k1, ymm\v, ymm\fixed"                "_mm256_cmpgt_epu16_mask";
    VPCMPNLEUW       "k, zmm, zmm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpcmpnleuw k1, zmm\v, zmm\fixed"                "_mm512_cmpgt_epu16_mask";
    VPCMPNLTB        "k, zmm, zmm"           "i8"   "i8"   avx512  avx512bw            evex  throughput r"vpcmpnltb k1, zmm\v, zmm\fixed"                 "_mm512_cmpge_epi8_mask";
    VPCMPNLTUB       "k, zmm, zmm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vpcmpnltub k1, zmm\v, zmm\fixed"                "_mm512_cmpge_epu8_mask";
    VPCMPNLTUD       "k, zmm, zmm"           "u32"  "u32"  avx512  avx512f             evex  throughput r"vpcmpnltud k1, zmm\v, zmm\fixed"                "_mm512_cmpge_epu32_mask";
    VPCMPNLTUQ       "k, zmm, zmm"           "u64"  "u64"  avx512  avx512f             evex  throughput r"vpcmpnltuq k1, zmm\v, zmm\fixed"                "_mm512_cmpge_epu64_mask";
    VPCMPNLTUW       "k, zmm, zmm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vpcmpnltuw k1, zmm\v, zmm\fixed"                "_mm512_cmpge_epu16_mask";
    VPCOMPRESSB      "ymm {k} {z}, ymm"      "u8"   "u8"   avx512  avx512_vbmi2        evex  chained    r"vpcompressb ymm\v {{k1}} {{z}}, ymm\v"          "_mm256_maskz_compress_epi8";
    VPCOMPRESSB      "zmm {k} {z}, zmm"      "u8"   "u8"   avx512  avx512_vbmi2        evex  chained    r"vpcompressb zmm\v {{k1}} {{z}}, zmm\v"          "_mm512_maskz_compress_epi8";
    VPCOMPRESSD      "m512 {k}, zmm"         "u32"  "u32"  avx512  avx512f             lines throughput r"vpcompressd zmmword ptr [\v] {{k1}}, zmm0"      "_mm512_mask_compressstoreu_epi32";
    VPCOMPRESSQ      "m512 {k}, zmm"         "u64"  "u64"  avx512  avx512f             lines throughput r"vpcompressq zmmword ptr [\v] {{k1}}, zmm0"      "_mm512_mask_compressstoreu_epi64";
    VPCOMPRESSQ      "zmm {k} {z}, zmm"      "u64"  "u64"  avx512  avx512f             evex  chained    r"vpcompressq zmm\v {{k1}} {{z}}, zmm\v"          "_mm512_maskz_compress_epi64";
    VPCOMPRESSW      "m256 {k}, ymm"         "u16"  "u16"  avx512  avx512_vbmi2        lines throughput r"vpcompressw ymmword ptr [\v] {{k1}}, ymm0"      "_mm256_mask_compressstoreu_epi16";
    VPCOMPRESSW      "m512 {k}, zmm"         "u16"  "u16"  avx512  avx512_vbmi2        lines throughput r"vpcompressw zmmword ptr [\v] {{k1}}, zmm0"      "_mm512_mask_compressstoreu_epi16";
    VPCOMPRESSW      "ymm {k} {z}, ymm"      "u16"  "u16"  avx512  avx512_vbmi2        evex  chained    r"vpcompressw ymm\v {{k1}} {{z}}, ymm\v"          "_mm256_maskz_compress_epi16";
    VPDPBUSD         "xmm, xmm, xmm"         "u8"   "i32"  avx512  avx512_vnni         evex  chained    r"{{evex}} vpdpbusd xmm\v, xmm\fixed, xmm\fixed"  "_mm_dpbusd_epi32";
    VPDPBUSD         "ymm, ymm, ymm"         "u8"   "i32"  avx512  avx512_vnni         evex  chained    r"{{evex}} vpdpbusd ymm\v, ymm\fixed, ymm\fixed"  "_mm256_dpbusd_epi32";
    VPDPBUSD         "zmm, zmm, zmm"         "u8"   "i32"  avx512  avx512_vnni         evex  chained    r"vpdpbusd zmm\v, zmm\fixed, zmm\fixed"           "_mm512_dpbusd_epi32";
    VPDPBUSDS        "zmm, zmm, zmm"         "u8"   "i32"  avx512  avx512_vnni         evex  chained    r"vpdpbusds zmm\v, zmm\fixed, zmm\fixed"          "_mm512_dpbusds_epi32";
    VPDPWSSD         "ymm, ymm, ymm"         "i16"  "i32"  avx512  avx512_vnni         evex  chained    r"{{evex}} vpdpwssd ymm\v, ymm\fixed, ymm\fixed"  "_mm256_dpwssd_epi32";
    VPDPWSSD         "zmm, zmm, zmm"         "i16"  "i32"  avx512  avx512_vnni         evex  chained    r"vpdpwssd zmm\v, zmm\fixed, zmm\fixed"           "_mm512_dpwssd_epi32";
    VPDPWSSDS        "zmm, zmm, zmm"         "i16"  "i32"  avx512  avx512_vnni         evex  chained    r"vpdpwssds zmm\v, zmm\fixed, zmm\fixed"          "_mm512_dpwssds_epi32";
    VPERMB           "xmm {k} {z}, xmm, xmm" "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermb xmm\v {{k1}} {{z}}, xmm\v, xmm\fixed"    "_mm_maskz_permutexvar_epi8";
    VPERMB           "xmm, xmm, xmm"         "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermb xmm\v, xmm\v, xmm\fixed"                 "_mm_permutexvar_epi8";
    VPERMB           "zmm {k} {z}, zmm, zmm" "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermb zmm\v {{k1}} {{z}}, zmm\v, zmm\fixed"    "_mm512_maskz_permutexvar_epi8";
    VPERMB           "zmm {k}, zmm, zmm"     "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermb zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_permutexvar_epi8";
    VPERMB           "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermb zmm\v, zmm\v, zmm\fixed"                 "_mm512_permutexvar_epi8";
    VPERMPD          "zmm, zmm, imm8"        "f64"  "f64"  avx512  avx512f             evex  chained    r"vpermpd zmm\v, zmm\v, 1"                        "_mm512_permutex_epi64";
    VPERMPD          "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vpermpd zmm\v, zmm\v, zmm\fixed"                "_mm512_permutexvar_epi64 _mm512_permutexvar_pd";
    VPERMPS          "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vpermps zmm\v, zmm\v, zmm\fixed"                "_mm512_permutexvar_epi32 _mm512_permutexvar_ps";
    VPERMQ           "zmm {k} {z}, zmm, zmm" "u64"  "u64"  avx512  avx512f             evex  chained    r"vpermq zmm\v {{k1}} {{z}}, zmm\v, zmm\fixed"    "_mm512_maskz_permutexvar_epi64";
    VPERMT2B         "zmm {k} {z}, zmm, zmm" "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermt2b zmm\v {{k1}} {{z}}, zmm\v, zmm\fixed"  "_mm512_maskz_permutex2var_epi8";
    VPERMT2B         "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpermt2b zmm\v, zmm\v, zmm\fixed"               "_mm512_permutex2var_epi8";
    VPERMT2D         "zmm, zmm, zmm"         "u32"  "u32"  avx512  avx512f             evex  chained    r"vpermt2d zmm\v, zmm\v, zmm\fixed"               "_mm512_permutex2var_epi32";
    VPERMT2PS        "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vpermt2ps zmm\v, zmm\v, zmm\fixed"              "_mm512_permutex2var_ps";
    VPERMT2Q         "zmm, zmm, zmm"         "u64"  "u64"  avx512  avx512f             evex  chained    r"vpermt2q zmm\v, zmm\v, zmm\fixed"               "_mm512_permutex2var_epi64";
    VPERMT2W         "ymm, ymm, ymm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpermt2w ymm\v, ymm\v, ymm\fixed"               "_mm256_permutex2var_epi16";
    VPERMT2W         "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpermt2w zmm\v, zmm\v, zmm\fixed"               "_mm512_permutex2var_epi16";
    VPERMW           "ymm, ymm, ymm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpermw ymm\v, ymm\v, ymm\fixed"                 "_mm256_permutexvar_epi16";
    VPERMW           "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpermw zmm\v, zmm\v, zmm\fixed"                 "_mm512_permutexvar_epi16";
    VPEXPANDB        "zmm {k} {z}, zmm"      "u8"   "u8"   avx512  avx512_vbmi2        evex  chained    r"vpexpandb zmm\v {{k1}} {{z}}, zmm\v"            "_mm512_maskz_expand_epi8";
    VPEXPANDB        "zmm {k}, zmm"          "u8"   "u8"   avx512  avx512_vbmi2        evex  chained    r"vpexpandb zmm\v {{k1}}, zmm\v"                  "_mm512_mask_expand_epi8";
    VPMADDUBSW       "zmm, zmm, zmm"         "u8"   "i16"  avx512  avx512bw            evex  chained    r"vpmaddubsw zmm\v, zmm\v, zmm\fixed"             "_mm512_maddubs_epi16";
    VPMADDWD         "zmm, zmm, zmm"         "i16"  "i32"  avx512  avx512bw            evex  chained    r"vpmaddwd zmm\v, zmm\v, zmm\fixed"               "_mm512_madd_epi16";
    VPMAXSD          "zmm {k}, zmm, zmm"     "i32"  "i32"  avx512  avx512f             evex  chained    r"vpmaxsd zmm\v {{k1}}, zmm\v, zmm\fixed"         "_mm512_mask_max_epi32";
    VPMAXSD          "zmm, zmm, zmm"         "i32"  "i32"  avx512  avx512f             evex  chained    r"vpmaxsd zmm\v, zmm\v, zmm\fixed"                "_mm512_max_epi32";
    VPMAXSQ          "xmm, xmm, xmm"         "i64"  "i64"  avx512  avx512vl            evex  chained    r"vpmaxsq xmm\v, xmm\v, xmm\fixed"                "_mm_max_epi64";
    VPMAXSQ          "ymm, ymm, ymm"         "i64"  "i64"  avx512  avx512vl            evex  chained    r"vpmaxsq ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epi64";
    VPMAXSQ          "zmm, zmm, zmm"         "i64"  "i64"  avx512  avx512f             evex  chained    r"vpmaxsq zmm\v, zmm\v, zmm\fixed"                "_mm512_max_epi64";
    VPMAXSW          "zmm, zmm, zmm"         "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpmaxsw zmm\v, zmm\v, zmm\fixed"                "_mm512_max_epi16";
    VPMAXUB          "zmm {k}, zmm, zmm"     "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpmaxub zmm\v {{k1}}, zmm\v, zmm\fixed"         "_mm512_mask_max_epu8";
    VPMAXUB          "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpmaxub zmm\v, zmm\v, zmm\fixed"                "_mm512_max_epu8";
    VPMAXUD          "zmm, zmm, zmm"         "u32"  "u32"  avx512  avx512f             evex  chained    r"vpmaxud zmm\v, zmm\v, zmm\fixed"                "_mm512_max_epu32";
    VPMAXUQ          "xmm, xmm, xmm"         "u64"  "u64"  avx512  avx512vl            evex  chained    r"vpmaxuq xmm\v, xmm\v, xmm\fixed"                "_mm_max_epu64";
    VPMAXUQ          "ymm, ymm, ymm"         "u64"  "u64"  avx512  avx512vl            evex  chained    r"vpmaxuq ymm\v, ymm\v, ymm\fixed"                "_mm256_max_epu64";
    VPMAXUQ          "zmm, zmm, zmm"         "u64"  "u64"  avx512  avx512f             evex  chained    r"vpmaxuq zmm\v, zmm\v, zmm\fixed"                "_mm512_max_epu64";
    VPMINSD          "zmm, zmm, zmm"         "i32"  "i32"  avx512  avx512f             evex  chained    r"vpminsd zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epi32";
    VPMINSQ          "xmm, xmm, xmm"         "i64"  "i64"  avx512  avx512vl            evex  chained    r"vpminsq xmm\v, xmm\v, xmm\fixed"                "_mm_min_epi64";
    VPMINSQ          "ymm, ymm, ymm"         "i64"  "i64"  avx512  avx512vl            evex  chained    r"vpminsq ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epi64";
    VPMINSQ          "zmm, zmm, zmm"         "i64"  "i64"  avx512  avx512f             evex  chained    r"vpminsq zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epi64";
    VPMINSW          "zmm, zmm, zmm"         "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpminsw zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epi16";
    VPMINUB          "zmm {k}, zmm, zmm"     "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpminub zmm\v {{k1}}, zmm\v, zmm\fixed"         "_mm512_mask_min_epu8";
    VPMINUB          "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpminub zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epu8";
    VPMINUD          "zmm, zmm, zmm"         "u32"  "u32"  avx512  avx512f             evex  chained    r"vpminud zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epu32";
    VPMINUQ          "xmm, xmm, xmm"         "u64"  "u64"  avx512  avx512vl            evex  chained    r"vpminuq xmm\v, xmm\v, xmm\fixed"                "_mm_min_epu64";
    VPMINUQ          "ymm, ymm, ymm"         "u64"  "u64"  avx512  avx512vl            evex  chained    r"vpminuq ymm\v, ymm\v, ymm\fixed"                "_mm256_min_epu64";
    VPMINUQ          "zmm, zmm, zmm"         "u64"  "u64"  avx512  avx512f             evex  chained    r"vpminuq zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epu64";
    VPMINUW          "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpminuw zmm\v, zmm\v, zmm\fixed"                "_mm512_min_epu16";
    VPMOVDB          "xmm, zmm"              "i32"  "i8"   avx512  avx512f             evex  chained    r"vpmovdb xmm\v, zmm\v"                           "_mm512_cvtepi32_epi8";
    VPMOVDW          "ymm, zmm"              "i32"  "i16"  avx512  avx512f             evex  chained    r"vpmovdw ymm\v, zmm\v"                           "_mm512_cvtepi32_epi16";
    VPMOVQB          "xmm, zmm"              "i64"  "i8"   avx512  avx512f             evex  chained    r"vpmovqb xmm\v, zmm\v"                           "_mm512_cvtepi64_epi8";
    VPMOVQD          "ymm, zmm"              "i64"  "i32"  avx512  avx512f             evex  chained    r"vpmovqd ymm\v, zmm\v"                           "_mm512_cvtepi64_epi32";
    VPMOVQW          "xmm, zmm"              "i64"  "i16"  avx512  avx512f             evex  chained    r"vpmovqw xmm\v, zmm\v"                           "_mm512_cvtepi64_epi16";
    VPMOVSDB         "xmm, zmm"              "i32"  "i8"   avx512  avx512f             evex  chained    r"vpmovsdb xmm\v, zmm\v"                          "_mm512_cvtsepi32_epi8";
    VPMOVSDW         "ymm, zmm"              "i32"  "i16"  avx512  avx512f             evex  chained    r"vpmovsdw ymm\v, zmm\v"                          "_mm512_cvtsepi32_epi16";
    VPMOVSXBD        "zmm, xmm"              "i8"   "i32"  avx512  avx512f             evex  chained    r"vpmovsxbd zmm\v, xmm\v"                         "_mm512_cvtepi8_epi32";
    VPMOVSXBQ        "zmm, xmm"              "i8"   "i64"  avx512  avx512f             evex  chained    r"vpmovsxbq zmm\v, xmm\v"                         "_mm512_cvtepi8_epi64";
    VPMOVSXBW        "zmm, ymm"              "i8"   "i16"  avx512  avx512bw            evex  chained    r"vpmovsxbw zmm\v, ymm\v"                         "_mm512_cvtepi8_epi16";
    VPMOVSXDQ        "zmm, ymm"              "i32"  "i64"  avx512  avx512f             evex  chained    r"vpmovsxdq zmm\v, ymm\v"                         "_mm512_cvtepi32_epi64";
    VPMOVSXWD        "zmm, ymm"              "i16"  "i32"  avx512  avx512f             evex  chained    r"vpmovsxwd zmm\v, ymm\v"                         "_mm512_cvtepi16_epi32";
    VPMOVSXWQ        "zmm, xmm"              "i16"  "i64"  avx512  avx512f             evex  chained    r"vpmovsxwq zmm\v, xmm\v"                         "_mm512_cvtepi16_epi64";
    VPMOVUSDB        "xmm, zmm"              "u32"  "u8"   avx512  avx512f             evex  chained    r"vpmovusdb xmm\v, zmm\v"                         "_mm512_cvtusepi32_epi8";
    VPMOVUSDW        "ymm, zmm"              "u32"  "u16"  avx512  avx512f             evex  chained    r"vpmovusdw ymm\v, zmm\v"                         "_mm512_cvtusepi32_epi16";
    VPMOVUSQB        "m64 {k}, zmm"          "u64"  "u8"   avx512  avx512f             lines throughput r"vpmovusqb qword ptr [\v] {{k1}}, zmm0"          "_mm512_mask_cvtusepi64_storeu_epi8";
    VPMOVWB          "xmm, ymm"              "i16"  "i8"   avx512  avx512bw            evex  chained    r"vpmovwb xmm\v, ymm\v"                           "_mm256_cvtepi16_epi8";
    VPMOVWB          "ymm, zmm"              "i16"  "i8"   avx512  avx512bw            evex  chained    r"vpmovwb ymm\v, zmm\v"                           "_mm512_cvtepi16_epi8";
    VPMOVZXBD        "zmm, xmm"              "u8"   "u32"  avx512  avx512f             evex  chained    r"vpmovzxbd zmm\v, xmm\v"                         "_mm512_cvtepu8_epi32";
    VPMOVZXBQ        "zmm, xmm"              "u8"   "u64"  avx512  avx512f             evex  chained    r"vpmovzxbq zmm\v, xmm\v"                         "_mm512_cvtepu8_epi64";
    VPMOVZXBW        "zmm, ymm"              "u8"   "u16"  avx512  avx512bw            evex  chained    r"vpmovzxbw zmm\v, ymm\v"                         "_mm512_cvtepu8_epi16";
    VPMOVZXDQ        "zmm, ymm"              "u32"  "u64"  avx512  avx512f             evex  chained    r"vpmovzxdq zmm\v, ymm\v"                         "_mm512_cvtepu32_epi64";
    VPMOVZXWD        "zmm, ymm"              "u16"  "u32"  avx512  avx512f             evex  chained    r"vpmovzxwd zmm\v, ymm\v"                         "_mm512_cvtepu16_epi32";
    VPMOVZXWQ        "zmm, xmm"              "u16"  "u64"  avx512  avx512f             evex  chained    r"vpmovzxwq zmm\v, xmm\v"                         "_mm512_cvtepu16_epi64";
    VPMULDQ          "zmm, zmm, zmm"         "i32"  "i64"  avx512  avx512f             evex  chained    r"vpmuldq zmm\v, zmm\v, zmm\fixed"                "_mm512_mul_epi32";
    VPMULLD          "zmm, zmm, zmm"         "i32"  "i32"  avx512  avx512f             evex  chained    r"vpmulld zmm\v, zmm\v, zmm\fixed"                "_mm512_mullo_epi32";
    VPMULLQ          "zmm, zmm, zmm"         "i64"  "i64"  avx512  avx512dq            evex  chained    r"vpmullq zmm\v, zmm\v, zmm\fixed"                "_mm512_mullo_epi64";
    VPMULTISHIFTQB   "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512vbmi          evex  chained    r"vpmultishiftqb zmm\v, zmm\v, zmm\fixed"         "_mm512_multishift_epi64_epi8";
    VPMULUDQ         "zmm, zmm, zmm"         "u32"  "u64"  avx512  avx512f             evex  chained    r"vpmuludq zmm\v, zmm\v, zmm\fixed"               "_mm512_mul_epu32";
    VPOPCNTQ         "zmm, zmm"              "u64"  "u64"  avx512  avx512_vpopcntdq    evex  chained    r"vpopcntq zmm\v, zmm\v"                          "_mm512_popcnt_epi64";
    VPROLD           "zmm, zmm, imm8"        "u32"  "u32"  avx512  avx512f             evex  chained    r"vprold zmm\v, zmm\v, 1"                         "_mm512_ror_epi32";
    VPSADBW          "zmm, zmm, zmm"         "u8"   "u16"  avx512  avx512bw            evex  chained    r"vpsadbw zmm\v, zmm\v, zmm\fixed"                "_mm512_sad_epu8";
    VPSHLDD          "zmm, zmm, zmm, imm8"   "u32"  "u32"  avx512  avx512_vbmi2        evex  chained    r"vpshldd zmm\v, zmm\v, zmm\fixed, 1"             "_mm512_shrdi_epi32";
    VPSHUFB          "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpshufb zmm\v, zmm\v, zmm\fixed"                "_mm512_shuffle_epi8";
    VPSLLVD          "zmm, zmm, zmm"         "u32"  "u32"  avx512  avx512f             evex  chained    r"vpsllvd zmm\v, zmm\v, zmm\fixed"                "_mm512_sllv_epi32";
    VPSLLVQ          "zmm, zmm, zmm"         "u64"  "u64"  avx512  avx512f             evex  chained    r"vpsllvq zmm\v, zmm\v, zmm\fixed"                "_mm512_sllv_epi64";
    VPSRAVD          "zmm, zmm, zmm"         "i32"  "i32"  avx512  avx512f             evex  chained    r"vpsravd zmm\v, zmm\v, zmm\fixed"                "_mm512_srav_epi32";
    VPSRLVD          "zmm, zmm, zmm"         "u32"  "u32"  avx512  avx512f             evex  chained    r"vpsrlvd zmm\v, zmm\v, zmm\fixed"                "_mm512_srlv_epi32";
    VPSRLVQ          "zmm, zmm, zmm"         "u64"  "u64"  avx512  avx512f             evex  chained    r"vpsrlvq zmm\v, zmm\v, zmm\fixed"                "_mm512_srlv_epi64";
    VPSRLVW          "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpsrlvw zmm\v, zmm\v, zmm\fixed"                "_mm512_srlv_epi16";
    VPSUBB           "zmm {k}, zmm, zmm"     "i8"   "i8"   avx512  avx512bw            evex  chained    r"vpsubb zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_sub_epi8";
    VPSUBB           "zmm, zmm, zmm"         "i8"   "i8"   avx512  avx512bw            evex  chained    r"vpsubb zmm\v, zmm\v, zmm\fixed"                 "_mm512_sub_epi8";
    VPSUBD           "zmm {k}, zmm, zmm"     "i32"  "i32"  avx512  avx512f             evex  chained    r"vpsubd zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_sub_epi32";
    VPSUBD           "zmm, zmm, zmm"         "i32"  "i32"  avx512  avx512f             evex  chained    r"vpsubd zmm\v, zmm\v, zmm\fixed"                 "_mm512_sub_epi32";
    VPSUBQ           "zmm, zmm, zmm"         "i64"  "i64"  avx512  avx512f             evex  chained    r"vpsubq zmm\v, zmm\v, zmm\fixed"                 "_mm512_sub_epi64";
    VPSUBUSB         "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpsubusb zmm\v, zmm\v, zmm\fixed"               "_mm512_subs_epu8";
    VPSUBW           "zmm {k}, zmm, zmm"     "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpsubw zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_sub_epi16";
    VPSUBW           "zmm, zmm, zmm"         "i16"  "i16"  avx512  avx512bw            evex  chained    r"vpsubw zmm\v, zmm\v, zmm\fixed"                 "_mm512_sub_epi16";
    VPTERNLOGD       "zmm, zmm, zmm, imm8"   "u32"  "u32"  avx512  avx512f             evex  chained    r"vpternlogd zmm\v, zmm\v, zmm\fixed, 0x96"       "_mm512_ternarylogic_epi32";
    VPTERNLOGQ       "zmm, zmm, zmm, imm8"   "u64"  "u64"  avx512  avx512f             evex  chained    r"vpternlogq zmm\v, zmm\v, zmm\fixed, 0x96"       "_mm512_ternarylogic_epi64";
    VPTESTMB         "k, zmm, zmm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vptestmb k1, zmm\v, zmm\fixed"                  "_mm512_test_epi8_mask";
    VPTESTMD         "k, ymm, ymm"           "u32"  "u32"  avx512  avx512vl            evex  throughput r"vptestmd k1, ymm\v, ymm\fixed"                  "_mm256_test_epi32_mask";
    VPTESTMD         "k, zmm, zmm"           "u32"  "u32"  avx512  avx512f             evex  throughput r"vptestmd k1, zmm\v, zmm\fixed"                  "_mm512_test_epi32_mask";
    VPTESTMQ         "k, zmm, zmm"           "u64"  "u64"  avx512  avx512f             evex  throughput r"vptestmq k1, zmm\v, zmm\fixed"                  "_mm512_test_epi64_mask";
    VPTESTMW         "k, zmm, zmm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vptestmw k1, zmm\v, zmm\fixed"                  "_mm512_test_epi16_mask";
    VPTESTNMB        "k, zmm, zmm"           "u8"   "u8"   avx512  avx512bw            evex  throughput r"vptestnmb k1, zmm\v, zmm\fixed"                 "_mm512_testn_epi8_mask";
    VPTESTNMD        "k, ymm, ymm"           "u32"  "u32"  avx512  avx512vl            evex  throughput r"vptestnmd k1, ymm\v, ymm\fixed"                 "_mm256_testn_epi32_mask";
    VPTESTNMD        "k, zmm, zmm"           "u32"  "u32"  avx512  avx512f             evex  throughput r"vptestnmd k1, zmm\v, zmm\fixed"                 "_mm512_testn_epi32_mask";
    VPTESTNMW        "k, ymm, ymm"           "u16"  "u16"  avx512  avx512bw            evex  throughput r"vptestnmw k1, ymm\v, ymm\fixed"                 "_mm256_testn_epi16_mask";
    VPUNPCKHBW       "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpunpckhbw zmm\v, zmm\v, zmm\fixed"             "_mm512_unpackhi_epi8";
    VPUNPCKHWD       "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpunpckhwd zmm\v, zmm\v, zmm\fixed"             "_mm512_unpackhi_epi16";
    VPUNPCKLBW       "zmm, zmm, zmm"         "u8"   "u8"   avx512  avx512bw            evex  chained    r"vpunpcklbw zmm\v, zmm\v, zmm\fixed"             "_mm512_unpacklo_epi8";
    VPUNPCKLWD       "zmm, zmm, zmm"         "u16"  "u16"  avx512  avx512bw            evex  chained    r"vpunpcklwd zmm\v, zmm\v, zmm\fixed"             "_mm512_unpacklo_epi16";
    VRNDSCALEPD      "zmm, zmm, imm8"        "f64"  "f64"  avx512  avx512f             evex  chained    r"vrndscalepd zmm\v, zmm\v, 0"                    "_mm512_roundscale_pd";
    VRNDSCALEPH      "zmm, zmm, imm8"        "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vrndscaleph zmm\v, zmm\v, 0"                    "_mm512_roundscale_ph";
    VRNDSCALEPS      "zmm, zmm, imm8"        "f32"  "f32"  avx512  avx512f             evex  chained    r"vrndscaleps zmm\v, zmm\v, 0"                    "_mm512_roundscale_ps";
    VRSQRT14PS       "zmm, zmm"              "f32"  "f32"  avx512  avx512f             evex  chained    r"vrsqrt14ps zmm\v, zmm\v"                        "_mm512_rsqrt14_ps";
    VRSQRT14SD       "xmm, xmm, xmm"         "f64"  "f64"  avx512  avx512vl            evex  chained    r"vrsqrt14sd xmm\v, xmm\fixed, xmm\v"             "_mm_rsqrt14_sd";
    VRSQRTSH         "xmm, xmm, xmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vrsqrtsh xmm\v, xmm\fixed, xmm\v"               "_mm_rsqrt_sh";
    VSCALEFPS        "xmm, xmm, xmm"         "f32"  "f32"  avx512  avx512vl            evex  chained    r"vscalefps xmm\v, xmm\v, xmm\fixed"              "_mm_scalef_ps";
    VSCALEFPS        "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vscalefps zmm\v, zmm\v, zmm\fixed"              "_mm512_scalef_ps";
    VSHUFF64X2       "zmm, zmm, zmm, imm8"   "f64"  "f64"  avx512  avx512f             evex  chained    r"vshuff64x2 zmm\v, zmm\v, zmm\fixed, 1"          "_mm512_shuffle_f32x4 _mm512_shuffle_f64x2";
    VSHUFI64X2       "zmm, zmm, zmm, imm8"   "u64"  "u64"  avx512  avx512f             evex  chained    r"vshufi64x2 zmm\v, zmm\v, zmm\fixed, 1"          "_mm512_broadcast_i32x4 _mm512_shuffle_i32x4 _mm512_shuffle_i64x2";
    VSHUFPD          "zmm, zmm, zmm, imm8"   "f64"  "f64"  avx512  avx512f             evex  chained    r"vshufpd zmm\v, zmm\v, zmm\fixed, 1"             "_mm512_permute_pd";
    VSHUFPS          "zmm, zmm, zmm, imm8"   "f32"  "f32"  avx512  avx512f             evex  chained    r"vshufps zmm\v, zmm\v, zmm\fixed, 1"             "_mm512_permute_ps _mm512_shuffle_epi32 _mm512_shuffle_ps";
    VSQRTPD          "ymm {k} {z}, ymm"      "f64"  "f64"  avx512  avx512vl            evex  chained    r"vsqrtpd ymm\v {{k1}} {{z}}, ymm\v"              "_mm256_maskz_sqrt_pd";
    VSQRTPD          "zmm {k} {z}, zmm"      "f64"  "f64"  avx512  avx512f             evex  chained    r"vsqrtpd zmm\v {{k1}} {{z}}, zmm\v"              "_mm512_maskz_sqrt_pd";
    VSQRTPD          "zmm, zmm"              "f64"  "f64"  avx512  avx512f             evex  chained    r"vsqrtpd zmm\v, zmm\v"                           "_mm512_sqrt_pd";
    VSQRTPS          "zmm {k} {z}, zmm"      "f32"  "f32"  avx512  avx512f             evex  chained    r"vsqrtps zmm\v {{k1}} {{z}}, zmm\v"              "_mm512_maskz_sqrt_ps";
    VSQRTPS          "zmm, zmm"              "f32"  "f32"  avx512  avx512f             evex  chained    r"vsqrtps zmm\v, zmm\v"                           "_mm512_sqrt_ps";
    VSQRTSH          "xmm, xmm, xmm"         "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vsqrtsh xmm\v, xmm\fixed, xmm\v"                "_mm_sqrt_sh";
    VSUBPD           "zmm {k} {z}, zmm, zmm" "f64"  "f64"  avx512  avx512f             evex  chained    r"vsubpd zmm\v {{k1}} {{z}}, zmm\v, zmm\fixed"    "_mm512_maskz_sub_pd";
    VSUBPD           "zmm {k}, zmm, zmm"     "f64"  "f64"  avx512  avx512f             evex  chained    r"vsubpd zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_sub_pd";
    VSUBPD           "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vsubpd zmm\v, zmm\v, zmm\fixed"                 "_mm512_sub_pd";
    VSUBPH           "zmm {k}, zmm, zmm"     "f16"  "f16"  avx512  avx512_fp16         evex  chained    r"vsubph zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_sub_ph";
    VSUBPS           "zmm {k} {z}, zmm, zmm" "f32"  "f32"  avx512  avx512f             evex  chained    r"vsubps zmm\v {{k1}} {{z}}, zmm\v, zmm\fixed"    "_mm512_maskz_sub_ps";
    VSUBPS           "zmm {k}, zmm, zmm"     "f32"  "f32"  avx512  avx512f             evex  chained    r"vsubps zmm\v {{k1}}, zmm\v, zmm\fixed"          "_mm512_mask_sub_ps";
    VSUBPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vsubps zmm\v, zmm\v, zmm\fixed"                 "_mm512_sub_ps";
    VUNPCKHPD        "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vunpckhpd zmm\v, zmm\v, zmm\fixed"              "_mm512_unpackhi_epi64 _mm512_unpackhi_pd";
    VUNPCKHPS        "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vunpckhps zmm\v, zmm\v, zmm\fixed"              "_mm512_unpackhi_epi32";
    VUNPCKLPD        "zmm, zmm, zmm"         "f64"  "f64"  avx512  avx512f             evex  chained    r"vunpcklpd zmm\v, zmm\v, zmm\fixed"              "_mm512_unpacklo_epi64 _mm512_unpacklo_pd";
    VUNPCKLPS        "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512f             evex  chained    r"vunpcklps zmm\v, zmm\v, zmm\fixed"              "_mm512_unpacklo_epi32";
    VXORPS           "zmm, zmm, zmm"         "f32"  "f32"  avx512  avx512dq            evex  chained    r"vxorps zmm\v, zmm\v, zmm\fixed"                 "_mm512_xor_pd _mm512_xor_ps _mm512_xor_si512";
    TDPBF16PS        "tmm, tmm, tmm"         "bf16" "f32"  amx     amx_bf16            tmm   chained    r"tdpbf16ps \v, \fixed"                           "_tile_dpbf16ps";
    TDPBSSD          "tmm, tmm, tmm"         "i8"   "i32"  amx     amx_int8            tmm   chained    r"tdpbssd \v, \fixed"                             "_tile_dpbssd";
    TDPBUSD          "tmm, tmm, tmm"         "u8"   "i32"  amx     amx_int8            tmm   chained    r"tdpbusd \v, \fixed"                             "_tile_dpbusd";
    TDPBUUD          "tmm, tmm, tmm"         "u8"   "u32"  amx     amx_int8            tmm   chained    r"tdpbuud \v, \fixed"                             "_tile_dpbuud";
    TILEZERO         "tmm"                   ""     ""     amx     amx_tile            tmm   throughput r"tilezero \v"                                    "_tile_zero";
    CLDEMOTE         "m8"                    ""     ""     atomics cldemote            lines throughput r"cldemote byte ptr [\v]"                         "_mm_cldemote";
    CMPXCHG          "lock m32, r32"         "u32"  "u32"  atomics lm                  lines chained    r"lock cmpxchg dword ptr [\v], eax"               "";
    CMPXCHG          "lock m64, r64"         "u64"  "u64"  atomics lm                  lines chained    r"lock cmpxchg qword ptr [\v], \v"                "";
    MFENCE           ""                      ""     ""     atomics sse2                lines throughput r"mfence"                                         "";
    OR               "lock m32, r32"         "u32"  "u32"  atomics lm                  lines chained    r"lock or dword ptr [\v], eax"                    "";
    OR               "lock m64, r64"         "u64"  "u64"  atomics lm                  lines chained    r"lock or qword ptr [\v], \fixed"                 "";
    PAUSE            ""                      ""     ""     atomics sse2                lines throughput r"pause"                                          "_mm_pause";
    UMONITOR         "r64"                   ""     ""     atomics waitpkg             lines throughput r"umonitor \v"                                    "_umonitor";
    XADD             "lock m32, r32"         "u32"  "u32"  atomics lm                  lines chained    r"lock xadd dword ptr [\v], eax"                  "";
    XADD             "lock m64, r64"         "u64"  "u64"  atomics lm                  lines chained    r"lock xadd qword ptr [\v], \fixed"               "";
    XCHG             "m32, r32"              "u32"  "u32"  atomics lm                  lines chained    r"xchg dword ptr [\v], eax"                       "";
    XCHG             "m64, r64"              "u64"  "u64"  atomics lm                  lines chained    r"xchg qword ptr [\v], \v"                        "";
    AND              "r32, r32"              "u32"  "u32"  scalar  lm                  r32   chained    r"and \v, \fixed"                                 "_kand_mask16";
    AND              "r64, r64"              "u64"  "u64"  scalar  lm                  r64   chained    r"and \v, \fixed"                                 "_kand_mask64";
    ANDN             "r32, r32, r32"         "u32"  "u32"  scalar  bmi1                r32   chained    r"andn \v, \v, \fixed"                            "_kandn_mask16";
    ANDN             "r64, r64, r64"         "u64"  "u64"  scalar  bmi1                r64   chained    r"andn \v, \v, \fixed"                            "_kandn_mask64";
    BLSR             "r64, r64"              "u64"  "u64"  scalar  bmi1                r64   chained    r"blsr \v, \v"                                    "_blsr_u64";
    BZHI             "r32, r32, r32"         "u32"  "u32"  scalar  bmi2                r32   chained    r"bzhi \v, \v, \fixed"                            "_bzhi_u32";
    BZHI             "r64, r64, r64"         "u64"  "u64"  scalar  bmi2                r64   chained    r"bzhi \v, \v, \fixed"                            "_bzhi_u64";
    LZCNT            "r32, r32"              "u32"  "u32"  scalar  abm                 r32   chained    r"lzcnt \v, \v"                                   "_lzcnt_u32";
    LZCNT            "r64, r64"              "u64"  "u64"  scalar  abm                 r64   chained    r"lzcnt \v, \v"                                   "_lzcnt_u64";
    NOT              "r64"                   "u64"  "u64"  scalar  lm                  r64   chained    r"not \v"                                         "_knot_mask64";
    OR               "r32, r32"              "u32"  "u32"  scalar  lm                  r32   chained    r"or \v, \fixed"                                  "_kor_mask16";
    OR               "r64, r64"              "u64"  "u64"  scalar  lm                  r64   chained    r"or \v, \fixed"                                  "_kor_mask64";
    PDEP             "r32, r32, r32"         "u32"  "u32"  scalar  bmi2                r32   chained    r"pdep \v, \v, \fixed"                            "_pdep_u32";
    PDEP             "r64, r64, r64"         "u64"  "u64"  scalar  bmi2                r64   chained    r"pdep \v, \v, \fixed"                            "_pdep_u64";
    PEXT             "r32, r32, r32"         "u32"  "u32"  scalar  bmi2                r32   chained    r"pext \v, \v, \fixed"                            "_pext_u32";
    PEXT             "r64, r64, r64"         "u64"  "u64"  scalar  bmi2                r64   chained    r"pext \v, \v, \fixed"                            "_pext_u64";
    POPCNT           "r32, r32"              "u32"  "u32"  scalar  popcnt              r32   chained    r"popcnt \v, \v"                                  "_mm_popcnt_u32";
    POPCNT           "r64, r64"              "u64"  "u64"  scalar  popcnt              r64   chained    r"popcnt \v, \v"                                  "_mm_popcnt_u64";
    RDTSC            ""                      "u64"  "u64"  scalar  tsc                 r64   throughput r"rdtsc"                                          "";
    TZCNT            "r32, r32"              "u32"  "u32"  scalar  bmi1                r32   chained    r"tzcnt \v, \v"                                   "_tzcnt_u32";
    TZCNT            "r64, r64"              "u64"  "u64"  scalar  bmi1                r64   chained    r"tzcnt \v, \v"                                   "_tzcnt_u64";
}

/// Forms no single step template can time.
pub const EXCEPTIONS: &[Form] = &[];
