extern crate ceptre;

#[macro_use]
extern crate criterion;

use criterion::Criterion;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("update/context1", |b| {
        let mut context1 = ceptre::Context::from_text(include_str!("wood.ceptre"));

        b.iter(|| {
            context1.append_state("#update");
            ceptre::update(&mut context1, |_: &ceptre::Phrase| None);
        })
    });

    c.bench_function("update/context2", |b| {
        let mut context2 = ceptre::Context::from_text(include_str!("spaceopera.ceptre"));

        b.iter(|| {
            ceptre::update(&mut context2, |_: &ceptre::Phrase| None);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
