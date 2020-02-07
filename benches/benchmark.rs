extern crate throne;

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
                    static ref CONTEXT: throne::Context =
                        throne::Context::from_text(include_str!("wood.throne")).with_test_rng();
                }

                CONTEXT.clone()
            },
            |mut context| {
                context.append_state("#update");
                throne::update(&mut context.core, |_: &throne::Phrase| None);
            },
        )
    });

    c.bench_function("update/context2", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: throne::Context =
                        throne::Context::from_text(include_str!("spaceopera.throne"))
                            .with_test_rng();
                }

                CONTEXT.clone()
            },
            |mut context| {
                throne::update(&mut context.core, |_: &throne::Phrase| None);
            },
        )
    });

    c.bench_function("update/context3", |b| {
        b.iter_with_setup(
            || {
                // only parse once, otherwise benchmark is affected
                lazy_static! {
                    static ref CONTEXT: throne::Context =
                        throne::Context::from_text(include_str!("increment.throne"))
                            .with_test_rng();
                }

                CONTEXT.clone()
            },
            |mut context| {
                context.append_state("#increment");
                throne::update(&mut context.core, |_: &throne::Phrase| None);
            },
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
