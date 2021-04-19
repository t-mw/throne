# throne

[![Crates.io][crates_img]][crates_link]

[crates_img]: https://img.shields.io/crates/v/throne.svg
[crates_link]: https://crates.io/crates/throne

A game scripting language for rapid prototyping and story logic:

```
// Declare the initial state as 'phrases', with one phrase per line.
Tom is child of David
Mary is sister of David
Sarah is child of Mary

// Define rules with the format: INPUT = OUTPUT.
CHILD is child of PARENT . AUNT is sister of PARENT . COUSIN is child of AUNT = COUSIN is cousin of CHILD

// The final state will be:
//    Mary is cousin of Tom
```

Rules are of the format `INPUT = OUTPUT`, where `INPUT` and `OUTPUT` are lists that use period (`.`) as a separator between items:
- `INPUT` is a list of one or more conditions that must pass for the rule to be executed. The conditions can either be state phrases that must exist or predicates (such as `>` using prefix notation) that must evaluate to true. Any matching state phrases are consumed by the rule on execution.
- `OUTPUT` is a list of state phrases that will be generated by the rule if it is executed.
- Identifiers in rules that use all capital letters (`CHILD`, `PARENT`, `AUNT` and `COUSIN` in the snippet above) are variables that will be assigned when the rule is executed.

Evaluating a throne script involves executing any rule that matches the current state until the set of matching rules is exhausted. Rules are executed in a random order and may be executed more than once.

### Build for wasm

1. Run `cargo install wasm-pack` to install [wasm-pack](https://github.com/rustwasm/wasm-pack).
1. Run `npm install ; npm start` in this directory.

### Examples
- [blocks](examples/blocks.throne): a simple tile matching game
- Used in [Urban Gift](https://twitter.com/UrbanGiftGame/)

### Design
Strongly influenced by https://www.cs.cmu.edu/~cmartens/ceptre.pdf.
