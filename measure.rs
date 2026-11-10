//! Times instruction forms on every core kind of the host and writes the median of `--runs` runs,
//! for the rows most runs settled, to `<output>/<isa>/<chip>/<extension>.csv`.
//!
//! Every form runs as two kernels, with `UNROLL` and with `2 × UNROLL` copies of the instruction
//! per loop iteration, so the difference of their times cancels the loop, the call and any mode
//! switch around it. Latency kernels chain every copy on the previous one, throughput kernels
//! rotate over a register class. Each pair is timed interleaved with a reference chain of adds in
//! its own clock domain, so no cycle counter is needed and clock changes cancel in the ratio.
//!
//! See also:
//! - <https://github.com/ashvardanian/less_slow.cpp>
//! - <https://github.com/ashvardanian/less_slow.rs>
//! - <https://ashvardanian.com/posts/cpu-ports/>
//! - <https://ashvardanian.com/posts/understanding-simd-complexity/>

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::ptr;
use std::time::{Duration, Instant};

use clap::Parser;

/// The relative accuracy the published rows claim, and the only chosen constant of the method.
pub const PRECISION: f64 = 0.01;

/// The spot fleet bills by the hour, so a whole pass over forms, core kinds and runs fits in one.
const BILLING_PERIOD: Duration = Duration::from_secs(3600);

const HEADER: &str = "mnemonic,form,input_type,output_type,core,mode,latency_cycles,\
                      throughput_cycles,nanoseconds,reference,reference_cycles,rounds,\
                      intrinsics,uops,ports";

/// Runs `iterations` loop iterations, each holding a fixed number of copies of one instruction.
pub type Kernel = fn(u64);

/// A chain of dependent adds that times one clock domain.
pub struct Reference {
    pub name: &'static str,
    /// Chains of `UNROLL` and `2 × UNROLL` adds.
    pub kernels: [Kernel; 2],
}

/// One cache line an atomic form works on, from a kernel's own stack, so it stays in L1.
///
/// Apple chips have 128-byte lines and Intel prefetches 64-byte lines in pairs, so 128 bytes keep
/// two lines from interfering on either.
#[derive(Clone, Copy, Default)]
#[repr(align(128))]
pub struct Line(#[expect(dead_code, reason = "kernels store the line's own address in it")] u64);

/// Where a form executes: the core's standard mode, or Arm streaming mode inside the SME unit.
#[derive(Clone, Copy)]
pub enum Mode {
    Standard,
    Streaming,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Standard => "standard",
            Mode::Streaming => "streaming",
        }
    }
}

/// One instruction form and the kernel pairs that time it.
pub struct Form {
    pub mnemonic: &'static str,
    pub form: &'static str,
    /// Type of the first source operand.
    pub input_type: &'static str,
    /// Type of the result, the input type unless the form widens or narrows.
    pub output_type: &'static str,
    /// Stem of the CSV file the row lands in.
    pub extension: &'static str,
    /// Linux `/proc/cpuinfo` flag that the form requires.
    pub feature: &'static str,
    pub mode: Mode,
    /// The chain in the form's clock domain that its kernels are timed against.
    pub reference: &'static Reference,
    /// Dependent chains of `UNROLL` and `2 × UNROLL` copies.
    pub latency: Option<[Kernel; 2]>,
    /// Copies rotating over the form's register class.
    pub throughput: Option<[Kernel; 2]>,
    /// Space-separated intrinsics that compile to the form.
    pub intrinsics: &'static str,
}

/// Counts the entries of an assembler list, where quoted entries may hold commas.
pub const fn registers(list: &str) -> usize {
    let bytes = list.as_bytes();
    let (mut count, mut quoted, mut index) = (1, false, 0);
    while index < bytes.len() {
        match bytes[index] {
            b'"' => quoted = !quoted,
            b',' if !quoted => count += 1,
            _ => {}
        }
        index += 1;
    }
    count
}

/// The least common multiple of `rotations`, so every throughput kernel's copies split evenly.
pub const fn unroll(rotations: &[usize]) -> usize {
    const fn divisor(mut left: usize, mut right: usize) -> usize {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }
    let (mut multiple, mut index) = (1, 0);
    while index < rotations.len() {
        multiple = multiple / divisor(multiple, rotations[index]) * rotations[index];
        index += 1;
    }
    multiple
}

/// Builds `UNROLL` and `FORMS` from one line per form, each line emitting both of its kernels.
///
/// The ISA module supplies `class!` with each register class's first, rotated and fixed registers,
/// `extension!` with each extension's file stem, mode and prologue, and `kernels!`, which wraps one
/// step template in a timed loop. In a template, `\v` is the rotated operand and `\fixed` the
/// class's reserved one. A `chained` line's latency kernel feeds `\v` back into itself; an
/// `independent` line's latency kernel also binds `\fixed` to that register, so it chains, while
/// its throughput kernel reads only the reserved one; a `throughput` line has no latency kernel.
/// In the `lines` class `\v` names a register that addresses a cache line holding its own address,
/// so a 64-bit load or atomic returns the next address.
macro_rules! forms {
    (@latency chained, $extension:ident, $class:ident, $template:literal) => {
        Some(kernels!($extension, $template, class!($class, first), class!($class, fixed)))
    };
    (@latency independent, $extension:ident, $class:ident, $template:literal) => {
        Some(kernels!($extension, $template, class!($class, first), class!($class, first)))
    };
    (@latency throughput, $extension:ident, $class:ident, $template:literal) => {
        None
    };
    ($(
        $mnemonic:ident $form:literal $input_type:literal $output_type:literal $extension:ident $feature:ident
        $class:ident $kernels:ident $template:literal $intrinsics:literal;
    )*) => {
        /// Copies per loop iteration in the shorter kernel of every pair.
        pub const UNROLL: usize = crate::unroll(&[$(crate::registers(class!($class, rotated))),*]);

        pub const FORMS: &[crate::Form] = &[$(
            crate::Form {
                mnemonic: stringify!($mnemonic),
                form: $form,
                input_type: $input_type,
                output_type: $output_type,
                extension: extension!($extension, stem),
                feature: stringify!($feature),
                mode: extension!($extension, mode),
                reference: extension!($extension, reference),
                latency: forms!(@latency $kernels, $extension, $class, $template),
                throughput: Some(kernels!(
                    $extension, $template, class!($class, rotated), class!($class, fixed)
                )),
                intrinsics: $intrinsics,
            },
        )*];
    };
}

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "aarch64")]
use arm as isa;
#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
use x86 as isa;
#[expect(
    dead_code,
    reason = "timed through WebAssembly runtimes rather than as native kernels"
)]
mod wasm;

/// Measures instruction latencies and throughputs on every core kind of this host.
#[derive(Parser)]
struct Arguments {
    /// Folder that receives `<isa>/<chip>/<extension>.csv`.
    #[arg(long, default_value = "data")]
    output: PathBuf,
    /// Chip folder name, by default the lowercased CPU brand string, like `apple-m5-pro`.
    #[arg(long)]
    chip: Option<String>,
    /// Passes over every form and core kind; each row is the median of the passes that settled.
    #[arg(long, default_value_t = 3)]
    runs: u32,
}

enum Error {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    /// The operating system did not report a property of the host.
    Unreported {
        property: String,
    },
    Pin {
        cpu: usize,
        source: io::Error,
    },
    Unsettled(Unsettled),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Error::Unreported { property } => {
                write!(formatter, "the host does not report {property}")
            }
            Error::Pin { cpu, source } => write!(formatter, "pinning to CPU {cpu}: {source}"),
            Error::Unsettled(unsettled) => write!(formatter, "{unsettled}"),
        }
    }
}

/// Logical CPUs of one microarchitecture on the host.
struct CoreKind {
    name: String,
    cpus: Range<usize>,
}

/// Where a doubling search stopped without two consecutive rounds agreeing.
struct Unsettled {
    what: &'static str,
    rounds: u32,
    previous: Option<f64>,
    last: Option<f64>,
}

impl fmt::Display for Unsettled {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let value = |value: Option<f64>| {
            value.map_or("no full round".to_owned(), |value| format!("{value:.4}"))
        };
        write!(
            formatter,
            "{} unsettled after {} rounds, the last two gave {} and {}",
            self.what,
            self.rounds,
            value(self.previous),
            value(self.last),
        )
    }
}

/// A value two consecutive rounds agreed on, with what the last round carried beside it.
struct Settled<T> {
    value: f64,
    carried: T,
    rounds: u32,
}

fn agree(previous: f64, last: f64) -> bool {
    previous > 0.0 && last > 0.0 && (last - previous).abs() <= PRECISION * last
}

/// Doubles the samples per round until two consecutive rounds agree within `PRECISION`, and gives
/// up when the next round would overrun `budget`.
fn converge<T>(
    what: &'static str,
    budget: Duration,
    mut round: impl FnMut(u64) -> Option<(f64, T)>,
) -> Result<Settled<T>, Unsettled> {
    let started = Instant::now();
    let (mut previous, mut rounds, mut samples) = (None, 0, 1);
    loop {
        let round_started = Instant::now();
        let result = round(samples);
        rounds += 1;
        let last = result.as_ref().map(|(value, _)| *value);
        if let (Some(previous), Some((value, carried))) = (previous, result) {
            if agree(previous, value) {
                return Ok(Settled {
                    value,
                    carried,
                    rounds,
                });
            }
        }
        if started.elapsed() + 2 * round_started.elapsed() > budget {
            return Err(Unsettled {
                what,
                rounds,
                previous,
                last,
            });
        }
        previous = last;
        samples *= 2;
    }
}

/// Smallest positive step between consecutive timer readings: the tick, or the cost of a reading
/// where that is larger.
fn timer_step(budget: Duration) -> Result<f64, Unsettled> {
    let steps = converge("timer step", budget, |steps| {
        let (mut smallest, mut seen, mut before) = (Duration::MAX, 0, Instant::now());
        while seen < steps {
            let now = Instant::now();
            if now > before {
                smallest = smallest.min(now - before);
                seen += 1;
            }
            before = now;
        }
        Some((smallest.as_nanos() as f64, ()))
    })?;
    Ok(steps.value)
}

fn time(kernel: Kernel, iterations: u64) -> f64 {
    let start = Instant::now();
    kernel(iterations);
    start.elapsed().as_nanos() as f64
}

/// Times one run, or returns `None` if the thread started or ended it outside `kind`.
fn time_on(kind: &CoreKind, kernel: Kernel, iterations: u64) -> Option<f64> {
    let started_on = current_cpu();
    let nanoseconds = time(kernel, iterations);
    let ended_on = current_cpu();
    (kind.cpus.contains(&started_on) && kind.cpus.contains(&ended_on)).then_some(nanoseconds)
}

/// A kernel pair and the loop count that makes a run of its shorter kernel last a run length.
#[derive(Clone, Copy)]
struct Pair {
    kernels: [Kernel; 2],
    iterations: u64,
}

impl Pair {
    fn calibrated(kernels: [Kernel; 2], run_nanoseconds: f64) -> Self {
        let mut iterations = 1;
        while time(kernels[0], iterations) < run_nanoseconds {
            iterations *= 2;
        }
        Pair {
            kernels,
            iterations,
        }
    }

    /// Nanoseconds per copy from the minima of the shorter and the longer kernel.
    fn per_copy(&self, short: f64, long: f64) -> f64 {
        (long - short) / (self.iterations as f64 * isa::UNROLL as f64)
    }
}

/// The form's time per copy over the reference's, and the reference's, from one round's minima.
fn ratio(minima: [f64; 4], form: Pair, reference: Pair) -> (f64, f64) {
    let reference_nanoseconds = reference.per_copy(minima[2], minima[3]);
    (
        form.per_copy(minima[0], minima[1]) / reference_nanoseconds,
        reference_nanoseconds,
    )
}

/// One core kind in one run, with what every form there is measured against.
struct Session<'a> {
    kind: &'a CoreKind,
    /// Nanoseconds a run of a shorter kernel lasts.
    run_nanoseconds: f64,
    /// Time each form may spend on its rounds.
    budget: Duration,
    /// Core cycles per copy of each reference measured so far.
    reference_cycles: Vec<(&'static str, f64)>,
}

impl Session<'_> {
    /// Times `form` interleaved with `reference`, round by round, until their ratio settles; carries
    /// the reference's nanoseconds per copy.
    fn gate(
        &self,
        what: &'static str,
        form: [Kernel; 2],
        reference: [Kernel; 2],
        budget: Duration,
    ) -> Result<Settled<f64>, Unsettled> {
        let form = Pair::calibrated(form, self.run_nanoseconds);
        let reference = Pair::calibrated(reference, self.run_nanoseconds);
        let runs = [
            (form.kernels[0], form.iterations),
            (form.kernels[1], form.iterations),
            (reference.kernels[0], reference.iterations),
            (reference.kernels[1], reference.iterations),
        ];
        converge(what, budget, |samples| {
            let mut minima = [f64::INFINITY; 4];
            for _ in 0..samples {
                for ((kernel, iterations), minimum) in runs.iter().zip(&mut minima) {
                    if let Some(nanoseconds) = time_on(self.kind, *kernel, *iterations) {
                        *minimum = minimum.min(nanoseconds);
                    }
                }
            }
            minima
                .iter()
                .all(|minimum| minimum.is_finite())
                .then(|| ratio(minima, form, reference))
        })
    }

    /// Core cycles per copy of `reference`: one for the core add chain, measured against it otherwise.
    fn reference_cycles(&mut self, reference: &'static Reference) -> Result<f64, Unsettled> {
        if ptr::eq(reference, &isa::CORE_REFERENCE) {
            return Ok(1.0);
        }
        if let Some((_, cycles)) = self
            .reference_cycles
            .iter()
            .find(|(name, _)| *name == reference.name)
        {
            return Ok(*cycles);
        }
        let core = isa::CORE_REFERENCE.kernels;
        let cycles = self
            .gate("reference", reference.kernels, core, self.budget)?
            .value;
        self.reference_cycles.push((reference.name, cycles));
        Ok(cycles)
    }

    fn measure(&mut self, form: &Form) -> Result<Measurement, Unsettled> {
        let reference_cycles = self.reference_cycles(form.reference)?;
        let gates = form.latency.iter().chain(&form.throughput).count() as u32;
        let budget = self.budget / gates;
        let reference = form.reference.kernels;
        let gate = |what, kernels: Option<[Kernel; 2]>| {
            kernels
                .map(|kernels| self.gate(what, kernels, reference, budget))
                .transpose()
        };
        let latency = gate("latency", form.latency)?;
        let throughput = gate("throughput", form.throughput)?;
        let windows = || latency.iter().chain(&throughput);
        let reference_nanoseconds =
            windows().map(|window| window.carried).sum::<f64>() / f64::from(gates);
        Ok(Measurement {
            latency_cycles: latency
                .as_ref()
                .map(|window| window.value * reference_cycles),
            throughput_cycles: throughput
                .as_ref()
                .map(|window| window.value * reference_cycles),
            nanoseconds: reference_nanoseconds / reference_cycles,
            reference_cycles,
            rounds: windows()
                .map(|window| window.rounds)
                .max()
                .unwrap_or_default(),
        })
    }
}

/// One form on one core kind, from one run or as the median of several.
#[derive(Clone, Copy)]
struct Measurement {
    latency_cycles: Option<f64>,
    throughput_cycles: Option<f64>,
    /// Nanoseconds per core cycle while the form was measured.
    nanoseconds: f64,
    /// Core cycles per copy of the form's reference.
    reference_cycles: f64,
    /// Rounds until the slower of the form's gates settled.
    rounds: u32,
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    match values.len() {
        0 => None,
        length if length % 2 == 1 => Some(values[middle]),
        _ => Some((values[middle - 1] + values[middle]) / 2.0),
    }
}

/// The median of every field over the runs that settled, unless fewer than most of `runs` did;
/// `rounds` is the most any run needed.
fn median_of(measurements: &[Measurement], runs: u32) -> Option<Measurement> {
    if measurements.len() as u32 * 2 <= runs {
        return None;
    }
    let field =
        |field: fn(&Measurement) -> Option<f64>| median(measurements.iter().filter_map(field));
    Some(Measurement {
        latency_cycles: field(|measurement| measurement.latency_cycles),
        throughput_cycles: field(|measurement| measurement.throughput_cycles),
        nanoseconds: field(|measurement| Some(measurement.nanoseconds))?,
        reference_cycles: field(|measurement| Some(measurement.reference_cycles))?,
        rounds: measurements
            .iter()
            .map(|measurement| measurement.rounds)
            .max()?,
    })
}

struct Row<'a> {
    form: &'a Form,
    core: &'a str,
    measurement: Measurement,
}

impl Row<'_> {
    fn csv(&self) -> String {
        let cycles = |value: Option<f64>| {
            value
                .map(|cycles| format!("{cycles:.3}"))
                .unwrap_or_default()
        };
        let (form, measurement) = (self.form, self.measurement);
        format!(
            "{},\"{}\",{},{},{},{},{},{},{:.4},{},{:.3},{},{},,",
            form.mnemonic,
            form.form,
            form.input_type,
            form.output_type,
            self.core,
            form.mode.name(),
            cycles(measurement.latency_cycles),
            cycles(measurement.throughput_cycles),
            measurement.nanoseconds,
            form.reference.name,
            measurement.reference_cycles,
            measurement.rounds,
            form.intrinsics,
        )
    }
}

/// Returns the open CSV file of `extension`, creating it with the header on first use.
fn file_for<'a>(
    files: &'a mut Vec<(&'static str, File)>,
    folder: &Path,
    extension: &'static str,
) -> Result<&'a mut File, Error> {
    let index = match files.iter().position(|(name, _)| *name == extension) {
        Some(index) => index,
        None => {
            let path = folder.join(format!("{extension}.csv"));
            let mut file = File::create(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            writeln!(file, "{HEADER}").map_err(|source| Error::Io { path, source })?;
            files.push((extension, file));
            files.len() - 1
        }
    };
    Ok(&mut files[index].1)
}

/// Lowercases a brand string into a folder name: "Apple M5 Pro" becomes `apple-m5-pro`.
fn folder_name(brand: &str) -> String {
    let lowercase = brand.to_lowercase();
    let words = lowercase.split(|character: char| !character.is_ascii_alphanumeric());
    words
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(target_os = "macos")]
fn sysctl_bytes(name: &str) -> Result<Vec<u8>, Error> {
    let unreported = || Error::Unreported {
        property: format!("sysctl {name}"),
    };
    let key = std::ffi::CString::new(name).map_err(|_| unreported())?;
    let query = |pointer: *mut libc::c_void, size: &mut usize| unsafe {
        libc::sysctlbyname(key.as_ptr(), pointer, size, ptr::null_mut(), 0)
    };
    let mut size = 0;
    if query(ptr::null_mut(), &mut size) != 0 {
        return Err(unreported());
    }
    let mut bytes = vec![0u8; size];
    if query(bytes.as_mut_ptr().cast(), &mut size) != 0 {
        return Err(unreported());
    }
    bytes.truncate(size);
    Ok(bytes)
}

#[cfg(target_os = "macos")]
fn sysctl_integer(name: &str) -> Result<i64, Error> {
    let bytes = sysctl_bytes(name)?;
    match <[u8; 4]>::try_from(bytes.as_slice()) {
        Ok(word) => Ok(i64::from(i32::from_ne_bytes(word))),
        Err(_) => <[u8; 8]>::try_from(bytes.as_slice())
            .map(i64::from_ne_bytes)
            .map_err(|_| Error::Unreported {
                property: format!("sysctl {name} as an integer"),
            }),
    }
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Result<String, Error> {
    let bytes = sysctl_bytes(name)?;
    let text = bytes.split(|byte| *byte == 0).next().unwrap_or_default();
    Ok(String::from_utf8_lossy(text).into_owned())
}

#[cfg(target_os = "macos")]
fn brand_string() -> Result<String, Error> {
    sysctl_string("machdep.cpu.brand_string")
}

#[cfg(target_os = "macos")]
fn supported(feature: &str) -> Result<bool, Error> {
    Ok(sysctl_integer(&format!("hw.optional.arm.{}", isa::apple_feature(feature)))? == 1)
}

#[cfg(target_os = "macos")]
fn core_kinds() -> Result<Vec<CoreKind>, Error> {
    let levels = sysctl_integer("hw.nperflevels")? as usize;
    let mut kinds = Vec::with_capacity(levels);
    let mut first_cpu = 0;
    // Logical CPU IDs count up from the slowest level, so the fastest level's IDs come last.
    for level in (0..levels).rev() {
        let count = sysctl_integer(&format!("hw.perflevel{level}.logicalcpu"))? as usize;
        let name = sysctl_string(&format!("hw.perflevel{level}.name"))?;
        kinds.push(CoreKind {
            name,
            cpus: first_cpu..first_cpu + count,
        });
        first_cpu += count;
    }
    kinds.reverse();
    Ok(kinds)
}

/// Steers the thread to `kind` with a QoS class, as macOS has no affinity API on Apple chips.
#[cfg(target_os = "macos")]
fn pin(kind: &CoreKind) -> Result<(), Error> {
    // Only background QoS keeps a thread on the slowest level, whose IDs start at 0.
    let class = match kind.cpus.start {
        0 => libc::qos_class_t::QOS_CLASS_BACKGROUND,
        _ => libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
    };
    match unsafe { libc::pthread_set_qos_class_self_np(class, 0) } {
        0 => Ok(()),
        status => Err(Error::Pin {
            cpu: kind.cpus.start,
            source: io::Error::from_raw_os_error(status),
        }),
    }
}

#[cfg(target_os = "macos")]
fn current_cpu() -> usize {
    let mut cpu = 0;
    unsafe { libc::pthread_cpu_number_np(&mut cpu) };
    cpu
}

#[cfg(target_os = "linux")]
fn cpuinfo_field(field: &str) -> Result<String, Error> {
    let path = PathBuf::from("/proc/cpuinfo");
    let cpuinfo = fs::read_to_string(&path).map_err(|source| Error::Io { path, source })?;
    let line = cpuinfo
        .lines()
        .find(|line| line.split(':').next().map(str::trim) == Some(field));
    let value = line
        .and_then(|line| line.split_once(':'))
        .map(|(_, value)| value.trim().to_owned());
    value.ok_or_else(|| Error::Unreported {
        property: format!("the /proc/cpuinfo {field:?} line"),
    })
}

#[cfg(target_os = "linux")]
fn brand_string() -> Result<String, Error> {
    cpuinfo_field("model name")
}

#[cfg(target_os = "linux")]
fn supported(feature: &str) -> Result<bool, Error> {
    let flags = cpuinfo_field("flags").or_else(|_| cpuinfo_field("Features"))?;
    Ok(flags.split_whitespace().any(|flag| flag == feature))
}

/// Parses the first range of a kernel CPU list such as `0-15,32-47`.
#[cfg(target_os = "linux")]
fn first_cpu_range(list: &str) -> Result<Range<usize>, Error> {
    let first = list.trim().split(',').next().unwrap_or_default();
    let (start, end) = first.split_once('-').unwrap_or((first, first));
    let parse = |text: &str| {
        text.parse::<usize>().map_err(|_| Error::Unreported {
            property: format!("a CPU range in {list:?}"),
        })
    };
    Ok(parse(start)?..parse(end)? + 1)
}

#[cfg(target_os = "linux")]
fn core_kinds() -> Result<Vec<CoreKind>, Error> {
    // Hybrid x86 parts list every core kind as a `cpu_<kind>` PMU; other hosts have one kind.
    let devices = PathBuf::from("/sys/devices");
    let entries = fs::read_dir(&devices).map_err(|source| Error::Io {
        path: devices,
        source,
    })?;
    let mut kinds = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name
            .to_str()
            .and_then(|name| name.strip_prefix("cpu_"))
        else {
            continue;
        };
        if let Ok(cpus) = fs::read_to_string(entry.path().join("cpus")) {
            kinds.push(CoreKind {
                name: name.to_owned(),
                cpus: first_cpu_range(&cpus)?,
            });
        }
    }
    if kinds.is_empty() {
        let path = PathBuf::from("/sys/devices/system/cpu/online");
        let online = fs::read_to_string(&path).map_err(|source| Error::Io { path, source })?;
        kinds.push(CoreKind {
            name: "core".to_owned(),
            cpus: first_cpu_range(&online)?,
        });
    }
    Ok(kinds)
}

#[cfg(target_os = "linux")]
fn pin(kind: &CoreKind) -> Result<(), Error> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::CPU_SET(kind.cpus.start, &mut set) };
    let size = std::mem::size_of::<libc::cpu_set_t>();
    match unsafe { libc::sched_setaffinity(0, size, &set) } {
        0 => Ok(()),
        _ => Err(Error::Pin {
            cpu: kind.cpus.start,
            source: io::Error::last_os_error(),
        }),
    }
}

#[cfg(target_os = "linux")]
fn current_cpu() -> usize {
    unsafe { libc::sched_getcpu() as usize }
}

fn run(arguments: Arguments) -> Result<(), Error> {
    let chip = match arguments.chip {
        Some(chip) => chip,
        None => folder_name(&brand_string()?),
    };
    let mut forms = Vec::new();
    for form in isa::FORMS.iter().chain(isa::EXCEPTIONS) {
        match supported(form.feature)? {
            true => forms.push(form),
            false => eprintln!(
                "skipped {} {}: no {}",
                form.mnemonic, form.form, form.feature
            ),
        }
    }
    let kinds = core_kinds()?;
    let runs = arguments.runs.max(1);
    let budget = BILLING_PERIOD / (forms.len() * kinds.len()) as u32 / runs;
    println!(
        "unroll {}, {} forms, {budget:.1?} per form",
        isa::UNROLL,
        forms.len()
    );
    let mut settled = vec![Vec::new(); kinds.len() * forms.len()];
    for run in 1..=runs {
        for (kind_index, kind) in kinds.iter().enumerate() {
            pin(kind)?;
            let step = timer_step(budget).map_err(Error::Unsettled)?;
            // Four minima enter every ratio, each off by up to one timer step.
            let run_nanoseconds = 4.0 * step / PRECISION;
            println!(
                "run {run} of {runs}, {} cores, CPUs {} to {}, timer step {step:.1} ns, runs of {:.1} us",
                kind.name,
                kind.cpus.start,
                kind.cpus.end - 1,
                run_nanoseconds / 1e3,
            );
            let mut session = Session {
                kind,
                run_nanoseconds,
                budget,
                reference_cycles: Vec::new(),
            };
            for (form_index, form) in forms.iter().enumerate() {
                match session.measure(form) {
                    Ok(measurement) => {
                        println!(
                            "{}",
                            Row {
                                form,
                                core: &kind.name,
                                measurement
                            }
                            .csv()
                        );
                        settled[kind_index * forms.len() + form_index].push(measurement);
                    }
                    Err(unsettled) => {
                        eprintln!(
                            "dropped {} {} on {}: {unsettled}",
                            form.mnemonic, form.form, kind.name
                        )
                    }
                }
            }
        }
    }
    let folder = arguments.output.join(isa::NAME).join(chip);
    fs::create_dir_all(&folder).map_err(|source| Error::Io {
        path: folder.clone(),
        source,
    })?;
    let mut files = Vec::new();
    for (kind_index, kind) in kinds.iter().enumerate() {
        for (form_index, form) in forms.iter().enumerate() {
            let measurements = &settled[kind_index * forms.len() + form_index];
            let Some(measurement) = median_of(measurements, runs) else {
                eprintln!(
                    "unstable {} {} on {}: settled in {} of {runs} runs",
                    form.mnemonic,
                    form.form,
                    kind.name,
                    measurements.len(),
                );
                continue;
            };
            let file = file_for(&mut files, &folder, form.extension)?;
            let line = Row {
                form,
                core: &kind.name,
                measurement,
            }
            .csv();
            let path = || folder.join(format!("{}.csv", form.extension));
            writeln!(file, "{line}").map_err(|source| Error::Io {
                path: path(),
                source,
            })?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Arguments::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("measure: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_divides_the_differences_and_settles_on_agreement() {
        fn idle(_: u64) {}
        let pair = |iterations| Pair {
            kernels: [idle, idle],
            iterations,
        };
        let copies = isa::UNROLL as f64;
        let (cycles, reference) = ratio(
            [100.0, 100.0 + 8.0 * copies, 50.0, 50.0 + copies],
            pair(2),
            pair(4),
        );
        assert_eq!((cycles, reference), (16.0, 0.25));
        assert!(agree(1.0, 1.0 + PRECISION / 2.0) && !agree(1.0, 1.0 + 2.0 * PRECISION));
        let mut rounds = [4.0, 2.0, 2.01].into_iter();
        let settled = converge("test", Duration::MAX, |_| {
            rounds.next().map(|value| (value, ()))
        })
        .ok();
        assert_eq!(
            settled.map(|settled| (settled.value, settled.rounds)),
            Some((2.01, 3))
        );
        assert_eq!(unroll(&[31, 4, 8, 2, 16]), 496);
        assert_eq!(registers(r#""za0h.s[w12, 0]", "za0h.s[w12, 1]", z0"#), 3);
    }

    #[test]
    fn median_needs_most_runs() {
        let run = |latency_cycles, rounds| Measurement {
            latency_cycles: Some(latency_cycles),
            throughput_cycles: None,
            nanoseconds: 0.25,
            reference_cycles: 1.0,
            rounds,
        };
        let median = median_of(&[run(4.0, 3), run(4.1, 9), run(3.0, 5)], 3).unwrap();
        assert_eq!(
            (
                median.latency_cycles,
                median.throughput_cycles,
                median.rounds
            ),
            (Some(4.0), None, 9)
        );
        assert_eq!(
            median_of(&[run(4.0, 3), run(3.0, 5)], 3)
                .unwrap()
                .latency_cycles,
            Some(3.5)
        );
        assert!(median_of(&[run(4.0, 3)], 3).is_none());
    }

    #[test]
    fn row_has_the_header_columns() {
        fn idle(_: u64) {}
        static REFERENCE: Reference = Reference {
            name: "add z.s",
            kernels: [idle, idle],
        };
        let form = Form {
            mnemonic: "FMOPA",
            form: "za.s, p/m, p/m, z.h, z.h",
            input_type: "f16",
            output_type: "f32",
            extension: "sme2",
            feature: "sme",
            mode: Mode::Streaming,
            reference: &REFERENCE,
            latency: Some([idle, idle]),
            throughput: None,
            intrinsics: "svmopa_za32_f16_m",
        };
        let measurement = Measurement {
            latency_cycles: Some(4.0),
            throughput_cycles: None,
            nanoseconds: 0.23334,
            reference_cycles: 3.0,
            rounds: 7,
        };
        let line = Row {
            form: &form,
            core: "Super",
            measurement,
        }
        .csv();
        assert_eq!(
            line,
            "FMOPA,\"za.s, p/m, p/m, z.h, z.h\",f16,f32,Super,streaming,4.000,,0.2333,add z.s,3.000,\
             7,svmopa_za32_f16_m,,"
        );
        let unquoted = line.replace("\"za.s, p/m, p/m, z.h, z.h\"", "form");
        assert_eq!(unquoted.split(',').count(), HEADER.split(',').count());
    }

    #[test]
    fn published_tables_start_with_the_header_and_keep_its_width() {
        // Commas inside quoted fields, like operand lists, don't separate columns.
        let columns = |line: &str| {
            line.split('"')
                .step_by(2)
                .map(|outside| outside.matches(',').count())
                .sum::<usize>()
                + 1
        };
        for isa in std::fs::read_dir("data").unwrap() {
            for chip in std::fs::read_dir(isa.unwrap().path()).unwrap() {
                for table in std::fs::read_dir(chip.unwrap().path()).unwrap() {
                    let path = table.unwrap().path();
                    let text = std::fs::read_to_string(&path).unwrap();
                    let mut lines = text.lines();
                    assert_eq!(lines.next(), Some(HEADER), "{}", path.display());
                    for line in lines {
                        assert_eq!(columns(line), columns(HEADER), "{}: {line}", path.display());
                    }
                }
            }
        }
    }
}
