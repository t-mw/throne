import initWasm, { Context, init } from "../pkg/throne.js";

async function main() {
  await initWasm();
  init();

  const text = `
at 0 0 wood . at 0 1 wood . at 1 1 wood . at 0 1 fire . #update
#update: {
  at X Y wood . at X Y fire = at X Y fire
  () = #spread
}
#spread . $at X Y fire . + X 1 X' . + Y' 1 Y = at X' Y fire . at X Y' fire
`;

  const context = Context.from_text(text);
  context.update();
  context.print();

  console.log(context.get_state());
}

main().catch(console.error);
