extern crate throne;

#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;

use criterion::Criterion;
use rand::{rngs::SmallRng, SeedableRng};

const TEST_SEED: u64 = 123;

fn build_context(text: &str) -> throne::Context {
    let mut rng = SmallRng::seed_from_u64(TEST_SEED);
    throne::ContextBuilder::new()
        .text(text)
        .rng(&mut rng)
        .build()
        .unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("update/context1", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: throne::Context =
                        build_context(include_str!("wood.throne"));
                }

                CONTEXT.clone()
            },
            |mut context| {
                context.push_state("#update");
                context
                    .update_with_side_input(|_: &throne::Phrase| None)
                    .unwrap();
            },
        )
    });

    c.bench_function("update/context2", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: throne::Context =
                        build_context(include_str!("spaceopera.throne"));
                }

                CONTEXT.clone()
            },
            |mut context| {
                context
                    .update_with_side_input(|_: &throne::Phrase| None)
                    .unwrap();
            },
        )
    });

    c.bench_function("update/context3", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: throne::Context =
                        build_context(include_str!("increment.throne"));
                }

                CONTEXT.clone()
            },
            |mut context| {
                context.push_state("#increment");
                context
                    .update_with_side_input(|_: &throne::Phrase| None)
                    .unwrap();
            },
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
