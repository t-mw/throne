#[macro_use]
extern crate lazy_static;
extern crate regex;

mod ceptre;

fn main() {
    println!("{:?}", ceptre::tokenize("test1 test2"));
}
