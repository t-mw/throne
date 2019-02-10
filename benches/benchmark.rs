extern crate ceptre;

#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;

use criterion::Criterion;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("update/context1", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: ceptre::Context =
                        ceptre::Context::from_text(include_str!("wood.ceptre")).with_test_rng();
                }

                CONTEXT.clone()
            },
            |mut context| {
                context.append_state("#update");
                ceptre::update(&mut context.core, |_: &[ceptre::Token]| None);
            },
        )
    });

    c.bench_function("update/context2", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: ceptre::Context =
                        ceptre::Context::from_text(include_str!("spaceopera.ceptre"))
                            .with_test_rng();
                }

                CONTEXT.clone()
            },
            |mut context| {
                ceptre::update(&mut context.core, |_: &[ceptre::Token]| None);
            },
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
