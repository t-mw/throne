#[macro_use]
extern crate lazy_static;

extern crate minifb;
extern crate pest;
extern crate rand;
#[macro_use]
extern crate pest_derive;
extern crate regex;

use minifb::{Key, KeyRepeat, Window, WindowOptions};

mod ceptre;

use std::io::Read;
use std::{fs, thread, time};

const WIDTH: i32 = 100;
const HEIGHT: i32 = 200;

fn main() {
    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH as usize,
        HEIGHT as usize,
        WindowOptions::default(),
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut context = ceptre::Context::from_text(include_str!("../blocks.ceptre"));

    let kd = context.str_to_atom("^kd");
    let ku = context.str_to_atom("^ku");
    let kp = context.str_to_atom("^kp");
    let left = context.str_to_atom("left");
    let right = context.str_to_atom("right");
    let up = context.str_to_atom("up");
    let down = context.str_to_atom("down");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        context.append_state("#tick");
        context.append_state("dt 3");
        context.print();

        let string_to_key = |s: &ceptre::Atom| match s {
            s if *s == left => Some(Key::Left),
            s if *s == right => Some(Key::Right),
            s if *s == up => Some(Key::Up),
            s if *s == down => Some(Key::Down),
            _ => None,
        };

        ceptre::update(&mut context, |p: &ceptre::Phrase| {
            if p.len() != 2 {
                return None;
            }

            match &p[0].string {
                a if *a == kd => string_to_key(&p[1].string).and_then(|k| {
                    if window.is_key_down(k) {
                        Some(p.clone())
                    } else {
                        None
                    }
                }),
                a if *a == ku => string_to_key(&p[1].string).and_then(|k| {
                    if !window.is_key_down(k) {
                        Some(p.clone())
                    } else {
                        None
                    }
                }),
                a if *a == kp => string_to_key(&p[1].string).and_then(|k| {
                    if window.is_key_pressed(k, KeyRepeat::Yes) {
                        Some(p.clone())
                    } else {
                        None
                    }
                }),
                _ => None,
            }
        });

        let mut buffer: Vec<u32> = vec![0; (WIDTH * HEIGHT) as usize];

        let is_valid_pos = |x, y| x < WIDTH && y < HEIGHT;

        for p in &context.state {
            match (
                p.get(0).and_then(|t| t.as_str(&context.string_cache)),
                p.get(2).and_then(|t| t.as_number()),
                p.get(3).and_then(|t| t.as_number()),
            ) {
                (Some("block-falling"), Some(x), Some(y))
                | (Some("block-set"), Some(x), Some(y)) => {
                    let color = 0x00b27474;

                    let x0 = x * 10;
                    let x1 = x0 + 10;

                    let y0 = y * 10;
                    let y1 = y0 + 10;

                    for y in y0..y1 {
                        for x in x0..x1 {
                            if is_valid_pos(x, y) {
                                let idx = x + WIDTH * (HEIGHT - 1 - y);
                                buffer[idx as usize] = color;
                            }
                        }
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
