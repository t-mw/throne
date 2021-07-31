# Throne

[![Crates.io](https://img.shields.io/crates/v/throne.svg)](https://crates.io/crates/throne)
[![Docs Status](https://docs.rs/throne/badge.svg)](https://docs.rs/throne)

A scripting language for game prototyping and story logic:

```
// Declare the initial state as 'phrases', with one phrase per line.
Mary is sister of David
Sarah is child of Mary
Tom is child of David

// Define rules with the format: INPUT = OUTPUT.
CHILD is child of PARENT . AUNT is sister of PARENT .
    COUSIN is child of AUNT = COUSIN is cousin of CHILD

// The final state will be:
//    Sarah is cousin of Tom
```

## Motivation

The inspiration for Throne comes from languages used to author interactive fiction, such as [Inform](https://inform7.com/).
As described in [this](https://brunodias.dev/2017/05/05/inform-prototyping.html) article, Inform can be used to prototype games from genres besides interactive fiction. By defining gameplay through rules, much of the verbosity of general purpose programming languages can be avoided. However, Inform and other interactive fiction authoring systems are too slow to execute their rules in every frame of a real-time gameplay loop and are difficult to embed in an existing engine.

Throne allows gameplay logic to be defined through rules and so provides some of the benefits of a rule-based language like Inform, but is also fast to execute and easy to embed in an existing engine. Its syntax is strongly influenced by the [Ceptre](https://www.cs.cmu.edu/~cmartens/ceptre.pdf) programming language.

## Examples
- [throne-playground](https://github.com/t-mw/throne-playground) - a web-based editor for Throne.
- [blocks](examples/blocks.throne) - a simple tile matching game run with `cargo run --example blocks`.
- [storylets-rs](https://github.com/t-mw/storylets-rs) - A storylet-based narrative engine for games.
- [Urban Gift](https://twitter.com/UrbanGiftGame/) - uses Throne for gameplay logic.

## Reference

Rules are of the format `INPUT = OUTPUT`, where `INPUT` and `OUTPUT` are lists that use period (`.`) as a separator between items:
- `INPUT` is a list of one or more conditions that must pass for the rule to be executed. The conditions can either be state phrases that must exist or predicates (such as `>` using prefix notation) that must evaluate to true. Any matching state phrases are consumed by the rule on execution.
- `OUTPUT` is a list of state phrases that will be generated by the rule if it is executed.
- Identifiers in rules that use all capital letters (`CHILD`, `PARENT`, `AUNT` and `COUSIN` in the snippet above) are variables that will be assigned when the rule is executed.

Evaluating a Throne script involves executing any rule that matches the current state until the set of matching rules is exhausted. Rules are executed in a random order and may be executed more than once.

### Predicates

The following predicates can be used as one of the items in a rule's list of inputs:

| Syntax | Effect | Example |
| --- | --- | --- |
| `+ X Y Z` | Matches when the sum of `X` and `Y` equals `Z` | `+ HEALTH 10 HEALED` |
| `- X Y Z` | Matches when the sum of `Y` and `Z` equals `X` | `- HEALTH 10 DAMAGED` |
| `< X Y` | Matches when `X` is less than `Y` | `< MONEY 0` |
| `> X Y` | Matches when `X` is greater than `Y` | `> MONEY 0` |
| `<= X Y` | Matches when `X` is less than or equal to `Y` | `<= MONEY 0` |
| `>= X Y` | Matches when `X` is greater than or equal to `Y` | `>= MONEY 0` |
| `% X Y Z` | Matches when the modulo of `X` with `Y` equals `Z` | `% DEGREES 360 REM` |
| `= X Y` | Matches when `X` equals `Y` | `= HEALTH 100` |
| `!X` | Matches when `X` does not exist in the state | `!this does not exist` |
| `^X` | Calls the host application and matches depending on the response | `^os-clock-hour 12` |

When a predicate accepts two input variables, both variables must be assigned a value for the predicate to produce a match. A value is assigned either by writing a constant inline or by sharing a variable with another of the rule's inputs.
When a predicate accepts three input variables and one of the variables remains unassigned, it will be assigned the expected value according to the effect of the predicate e.g. `A` will be assigned the value `8` in `+ 2 A 10`.

### Constructs

Special syntax exists to make it easier to write complex rules, but in the end these constructs compile down to the simple form described in the introduction. The following table lists the available constructs:

| Syntax | Effect | Example | Compiled Form |
| --- | --- | --- | --- |
| Input phrase prefixed with `$` | Copies the input phrase to the rule output. | `$foo = bar` | `foo = bar . foo` |
| A set of rules surrounded by curly braces prefixed with `INPUT:` where `INPUT` is a list of phrases | Copies `INPUT` to each rule's inputs. | <pre>foo . bar: {<br/>  hello = world<br/>  123 = 456<br/>}</pre> | <pre>foo . bar . hello = world<br/>foo . bar . 123 = 456</pre> |
| `<<PHRASE . PHRASES` where `PHRASE` is a single phrase and `PHRASES` is a list of phrases | Replaces `<<PHRASE` with `PHRASES` wherever it exists in a rule's list of inputs. | <pre><<arithmetic A B . + A 1 B<br/><<arithmetic A B . - A 1 B<br/>foo . <<arithmetic X Y = Y</pre> | <pre>foo . + X 1 Y = Y<br/>foo . - X 1 Y = Y</pre> |

### The `()` Phrase

The `()` phrase represents the absence of state. When present in a rule's list of outputs it has no effect, besides making it possible to write rules that produce no output (e.g. `foo = ()`).
When present in a rule's list of inputs it has the effect of producing a match when no other rules can be matched. For example, in the following script the first rule will only ever be matched last:

```
foo           // initial state
() . bar = () // matched last
foo = bar     // matched first
```

In this way `()` can be used as a form of control flow, overriding the usual random order of rule execution.

### Stage Phrases

Prefixing a phrase with `#` marks it as a 'stage' phrase.
Stage phrases behave in largely the same way as normal phrases, but their presence should be used to indicate how far a script has executed within a sequence of 'stages'. A stage phrase can be included in a rule's inputs to only execute the rule within that stage of the script's execution, and included in a rule's outputs to define the transition to a new stage of the script's execution.

Stage phrases only differ in their behavior to normal phrases when used as a prefix to a set of curly braces. In this case the stage phrase will be copied to not only the inputs of the rules within the braces, but also the outputs, except when a rule includes `()` as an input phrase. This makes it easy to scope execution of the prefixed set of rules to a stage and finally transition to a second stage once execution of the first stage is complete.

| Example | Compiled Form |
| --- | --- |
| <pre>#first-stage: {<br/>  foo = bar<br/>  () = #second-stage<br/>}</pre> | <pre>#first-stage . foo = #first-stage . bar<br/>#first-stage . () = #second-stage</pre> |

## Build for Wasm

1. Run `cargo install wasm-pack` to install [wasm-pack](https://github.com/rustwasm/wasm-pack).
1. Run `npm install ; npm start` in this directory.
