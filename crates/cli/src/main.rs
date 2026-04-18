fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <a> <b>", args[0]);
        std::process::exit(1);
    }
    let a: i32 = args[1].parse().expect("First argument must be an integer");
    let b: i32 = args[2].parse().expect("Second argument must be an integer");
    let result = core::add(a, b);
    println!("{a} + {b} = {result}");
}
