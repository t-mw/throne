#[macro_use]
extern crate lazy_static;

extern crate minifb;
extern crate rand;
extern crate regex;

use minifb::{Key, Window, WindowOptions};

mod ceptre;

use std::io::Read;
use std::{fs, thread, time};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() {
    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut file = fs::File::open("test.ceptre").expect("file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("read_to_string");
    let mut context = ceptre::Context::from_text(&contents);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        context.append_state("#update");

        ceptre::update(&mut context, |p: &ceptre::Phrase| {
            if p.len() != 2 {
                return None;
            }

            match p[0].string.as_str() {
                "!kd" => {
                    let key = match p[1].string.as_str() {
                        "right" => Some(Key::Right),
                        _ => None,
                    };

                    key.and_then(|k| {
                        if window.is_key_down(k) {
                            Some(p.clone())
                        } else {
                            None
                        }
                    })
                }
                "!ku" => {
                    let key = match p[1].string.as_str() {
                        "right" => Some(Key::Right),
                        _ => None,
                    };

                    key.and_then(|k| {
                        if !window.is_key_down(k) {
                            Some(p.clone())
                        } else {
                            None
                        }
                    })
                }
                _ => None,
            }
        });

        let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

        let is_valid_pos = |x, y| x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT;

        for p in context.state.iter() {
            use std::str::FromStr;

            match (
                p.get(0).map(|t| t.string.as_str()),
                p.get(1).map(|t| t.string.as_str()),
                p.get(2).map(|t| t.string.as_str()),
                p.get(3).map(|t| t.string.as_str()),
            ) {
                (Some("at"), Some(x), Some(y), Some("fire")) => {
                    let x = usize::from_str(x);
                    let y = usize::from_str(y);

                    let color = 0x00b27474;

                    match (x, y) {
                        (Ok(x), Ok(y)) => {
                            let x0 = x * 10;
                            let x1 = x0 + 10;

                            let y0 = y * 10;
                            let y1 = y0 + 10;

                            for y in y0..y1 {
                                for x in x0..x1 {
                                    if is_valid_pos(x, y) {
                                        let idx = x + WIDTH * y;
                                        buffer[idx] = color;
                                    }
                                }
                            }
                        }
                        _ => (),
                    }
                }
                _ => (),
            }
        }

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer).unwrap();

        thread::sleep(time::Duration::from_millis(33));
    }
}
