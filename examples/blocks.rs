#[cfg(not(target_arch = "wasm32"))]
fn main() {
    example::main();
}

#[cfg(target_arch = "wasm32")]
fn main() {
    println!("This example is not supported on WebAssembly");
}

#[cfg(not(target_arch = "wasm32"))]
mod example {
    extern crate lazy_static;

    extern crate minifb;
    extern crate pest;
    extern crate pest_derive;
    extern crate rand;
    extern crate regex;

    use minifb::{Key, KeyRepeat, Window, WindowOptions};

    use std::{thread, time};

    const WIDTH: usize = 100;
    const HEIGHT: usize = 200;

    pub fn main() {
        let mut window = Window::new(
            "Test - ESC to exit",
            WIDTH,
            HEIGHT,
            WindowOptions::default(),
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

        let mut context = throne::ContextBuilder::new()
            .text(include_str!("blocks.throne"))
            .build()
            .unwrap_or_else(|e| panic!("{}", e));

        let kd = context.str_to_atom("key-down");
        let ku = context.str_to_atom("key-up");
        let kp = context.str_to_atom("key-pressed");
        let left = context.str_to_atom("left");
        let right = context.str_to_atom("right");
        let up = context.str_to_atom("up");
        let down = context.str_to_atom("down");

        while window.is_open() && !window.is_key_down(Key::Escape) {
            context.push_state("#update");
            // context.print();

            let string_to_key = |s: &throne::Atom| match s {
                s if *s == left => Some(Key::Left),
                s if *s == right => Some(Key::Right),
                s if *s == up => Some(Key::Up),
                s if *s == down => Some(Key::Down),
                _ => None,
            };

            context
                .update_with_side_input(|p: &throne::Phrase| {
                    if p.len() != 2 {
                        return None;
                    }

                    match &p[0].atom {
                        a if *a == kd => string_to_key(&p[1].atom).and_then(|k| {
                            if window.is_key_down(k) {
                                Some(p.to_vec())
                            } else {
                                None
                            }
                        }),
                        a if *a == ku => string_to_key(&p[1].atom).and_then(|k| {
                            if !window.is_key_down(k) {
                                Some(p.to_vec())
                            } else {
                                None
                            }
                        }),
                        a if *a == kp => string_to_key(&p[1].atom).and_then(|k| {
                            if window.is_key_pressed(k, KeyRepeat::Yes) {
                                Some(p.to_vec())
                            } else {
                                None
                            }
                        }),
                        _ => None,
                    }
                })
                .unwrap();

            let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

            let is_valid_pos = |x, y| x < WIDTH as i32 && y < HEIGHT as i32;

            for phrase_id in context.core.state.iter() {
                let p = context.core.state.get(phrase_id);

                match (
                    p.first().and_then(|t| t.as_str(&context.string_cache)),
                    p.get(2).and_then(|t| t.as_integer()),
                    p.get(3).and_then(|t| t.as_integer()),
                    p.get(4).and_then(|t| t.as_integer()),
                ) {
                    (Some("block-falling"), Some(x), Some(y), _)
                    | (Some("block-set"), _, Some(x), Some(y)) => {
                        let color = 0x00b27474;

                        let x0 = x * 10;
                        let x1 = x0 + 10;

                        let y0 = y * 10;
                        let y1 = y0 + 10;

                        for y in y0..y1 {
                            for x in x0..x1 {
                                if is_valid_pos(x, y) {
                                    let idx = x + WIDTH as i32 * (HEIGHT as i32 - 1 - y);
                                    buffer[idx as usize] = color;
                                }
                            }
                        }
                    }
                    _ => (),
                }
            }

            // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
            window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

            thread::sleep(time::Duration::from_millis(33));
        }
    }
}
