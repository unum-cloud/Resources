//! WebAssembly SIMD opcodes as WAT text, for modules timed under each runtime.
//!
//! See also:
//! - <https://ashvardanian.com/posts/understanding-simd-complexity/>

/// One opcode, written as the step that feeds its result back into `$value`.
pub struct Opcode {
    pub opcode: &'static str,
    pub input_type: &'static str,
    pub output_type: &'static str,
    /// Expression over `$value`, the chained local, and `$other`, a loop-invariant local.
    pub step: &'static str,
}

pub const OPCODES: &[Opcode] = &[
    Opcode {
        opcode: "i32x4.add",
        input_type: "i32",
        output_type: "i32",
        step: "(i32x4.add (local.get $value) (local.get $other))",
    },
    Opcode {
        opcode: "f32x4.mul",
        input_type: "f32",
        output_type: "f32",
        step: "(f32x4.mul (local.get $value) (local.get $other))",
    },
    Opcode {
        opcode: "f32x4.relaxed_madd",
        input_type: "f32",
        output_type: "f32",
        step: "(f32x4.relaxed_madd (local.get $value) (local.get $other) (local.get $other))",
    },
    Opcode {
        opcode: "i32x4.dot_i16x8_s",
        input_type: "i16",
        output_type: "i32",
        step: "(i32x4.dot_i16x8_s (local.get $value) (local.get $other))",
    },
    Opcode {
        opcode: "i8x16.swizzle",
        input_type: "u8",
        output_type: "u8",
        step: "(i8x16.swizzle (local.get $other) (local.get $value))",
    },
    Opcode {
        opcode: "i8x16.popcnt",
        input_type: "u8",
        output_type: "u8",
        step: "(i8x16.popcnt (local.get $value))",
    },
];

/// A module exporting `run(iterations)`, whose loop body holds `copies` steps rotating over
/// `streams` independent locals.
pub fn module(opcode: &Opcode, copies: usize, streams: usize) -> String {
    let locals: String = (0..streams)
        .map(|stream| format!(" (local $value{stream} v128)"))
        .collect();
    let body: String = (0..copies)
        .map(|copy| {
            let value = format!("$value{}", copy % streams);
            format!(
                "      (local.set {value} {})\n",
                opcode.step.replace("$value", &value)
            )
        })
        .collect();
    format!(
        "(module\n  (func (export \"run\") (param $iterations i64){locals} (local $other v128)\n    \
         (local.set $other (v128.const i32x4 1 1 1 1))\n    (loop $repeat\n{body}      \
         (local.set $iterations (i64.sub (local.get $iterations) (i64.const 1)))\n      \
         (br_if $repeat (i64.ne (local.get $iterations) (i64.const 0))))))\n"
    )
}
