extern crate ceptre;

#[macro_use]
extern crate criterion;

use criterion::Criterion;

fn criterion_benchmark(c: &mut Criterion) {
    let mut context = ceptre::Context::from_text("\
at 1 1 wood . at 1 1 wood . at 1 1 wood . at 1 1 wood . at 1 1 wood\n\
\n\
#update:\n\
  at X Y wood . + X 1 X' . + Y 1 Y' . + X'' 1 X . + Y'' 1 Y = at X' Y' fire . at X' Y'' fire . at X'' Y' fire . at X'' Y'' fire\n\
  at X Y fire . + X 1 X' = at' X' Y fire\n\
  at X Y fire . + Y 1 Y' = at' X Y' fire\n\
  () = #process\n\
\n\
#process:\n\
  at' X Y I = at X Y I\n\
  () =
");

    c.bench_function("update", |b| {
        b.iter(|| {
            context.append_state("#update");
            ceptre::update(&mut context, |_: &ceptre::Phrase| None);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
