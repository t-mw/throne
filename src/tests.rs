use crate::matching::*;
use crate::parser;
use crate::rule::{LineColSpan, Rule, RuleBuilder};
use crate::state::{self, State};
use crate::string_cache::{Atom, StringCache};
use crate::token::*;
use crate::update;
use crate::Context;

#[cfg(not(target_arch = "wasm32"))]
use pretty_assertions::{assert_eq, assert_ne};
use rand::{rngs::SmallRng, SeedableRng};

#[test]
fn context_from_text_empty_test() {
    let context = Context::from_text("").unwrap();
    assert!(context.core.state.get_all().is_empty());
}

#[test]
fn context_from_text_unicode_test() {
    // ñ, black square, green heart, scottish flag
    let mut context = Context::from_text("\"ñ◼️💚🏴󠁧󠁢󠁳󠁣󠁴󠁿\"").unwrap();
    assert_eq!(
        context.core.state.get_all(),
        [tokenize("\"ñ◼️💚🏴󠁧󠁢󠁳󠁣󠁴󠁿\"", &mut context.string_cache)]
    );
}

#[test]
fn context_from_text_state_test() {
    let mut context = Context::from_text(
        "at 0 0 wood . at 0 0 wood . #update \n\
             at 1 2 wood",
    )
    .unwrap();

    assert_eq!(
        context.core.state.get_all(),
        [
            tokenize("at 0 0 wood", &mut context.string_cache),
            tokenize("at 0 0 wood", &mut context.string_cache),
            tokenize("#update", &mut context.string_cache),
            tokenize("at 1 2 wood", &mut context.string_cache),
        ]
    );
}

#[test]
fn context_from_text_prefix_test() {
    let mut context = Context::from_text(
        "at 0 0 wood . at 1 2 wood = at 1 0 wood \n\
             asdf . #test: { \n\
             at 3 4 wood = at 5 6 wood . at 7 8 wood \n\
             }",
    )
    .unwrap();

    context.print();

    assert_eq!(
        context.core.rules,
        [
            RuleBuilder::new(
                vec![
                    tokenize("at 0 0 wood", &mut context.string_cache),
                    tokenize("at 1 2 wood", &mut context.string_cache),
                ],
                vec![tokenize("at 1 0 wood", &mut context.string_cache)],
                LineColSpan {
                    line_start: 1,
                    line_end: 1,
                    col_start: 1,
                    col_end: 41
                }
            )
            .build(0),
            RuleBuilder::new(
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("asdf", &mut context.string_cache),
                    tokenize("at 3 4 wood", &mut context.string_cache),
                ],
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("at 5 6 wood", &mut context.string_cache),
                    tokenize("at 7 8 wood", &mut context.string_cache),
                ],
                LineColSpan {
                    line_start: 3,
                    line_end: 3,
                    col_start: 1,
                    col_end: 41
                }
            )
            .build(1),
            // automatically added rule to avoid infinite loops
            RuleBuilder::new(
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("asdf", &mut context.string_cache),
                    tokenize(parser::QUI, &mut context.string_cache),
                ],
                vec![],
                LineColSpan {
                    line_start: 2,
                    line_end: 4,
                    col_start: 1,
                    col_end: 2
                }
            )
            .build(2)
        ]
    );
}

#[test]
fn context_from_text_copy_test() {
    let mut context = Context::from_text("at 0 0 wood . $at 1 2 wood = at 1 0 wood").unwrap();

    assert_eq!(
        context.core.rules,
        [RuleBuilder::new(
            vec![
                tokenize("at 0 0 wood", &mut context.string_cache),
                tokenize("at 1 2 wood", &mut context.string_cache),
            ],
            vec![
                tokenize("at 1 2 wood", &mut context.string_cache),
                tokenize("at 1 0 wood", &mut context.string_cache)
            ],
            LineColSpan {
                line_start: 1,
                line_end: 1,
                col_start: 1,
                col_end: 41
            }
        )
        .build(0)]
    );
}

#[test]
fn context_from_text_rules_newline_test() {
    let mut context = Context::from_text(
        "broken line 1 =\n\
             broken line 2 .\n\
             broken line 3 \n\
             . broken line 4\n\
             text\n\
             = \"broken\ntext\"",
    )
    .unwrap();

    assert_eq!(
        context.core.rules,
        [
            RuleBuilder::new(
                vec![tokenize("broken line 1", &mut context.string_cache)],
                vec![
                    tokenize("broken line 2", &mut context.string_cache),
                    tokenize("broken line 3", &mut context.string_cache),
                    tokenize("broken line 4", &mut context.string_cache),
                ],
                LineColSpan {
                    line_start: 1,
                    line_end: 4,
                    col_start: 1,
                    col_end: 16
                }
            )
            .build(0),
            RuleBuilder::new(
                vec![tokenize("text", &mut context.string_cache)],
                vec![tokenize("\"broken\ntext\"", &mut context.string_cache)],
                LineColSpan {
                    line_start: 5,
                    line_end: 7,
                    col_start: 1,
                    col_end: 6
                }
            )
            .build(1)
        ]
    );
}

#[test]
fn context_from_text_backwards_predicate_simple_test() {
    let mut rng = test_rng();
    let mut context = Context::from_text_rng(
        "<<back1 C . ?state1 C D\n\
             <<back2 E F . ?state2 E F\n\
             <<back1 A . test . <<back2 B A = ()",
        &mut rng,
    )
    .unwrap();
    // intentionally separate backwards predicates by simple 'test' atom to test parsing

    context.print();

    let replacement_atom = context.core.rules[0].inputs[0][2].atom;
    let replacement = context
        .string_cache
        .atom_to_str(replacement_atom)
        .unwrap()
        .to_string();
    assert!(replacement.starts_with("D_BACK"));

    assert_eq!(
        context.core.rules,
        [RuleBuilder::new(
            vec![
                tokenize(
                    &format!("?state1 A {}", replacement),
                    &mut context.string_cache
                ),
                tokenize("test", &mut context.string_cache),
                tokenize("?state2 B A", &mut context.string_cache)
            ],
            vec![],
            LineColSpan {
                line_start: 3,
                line_end: 3,
                col_start: 1,
                col_end: 37
            }
        )
        .build(0)]
    );
}

#[test]
fn context_from_text_backwards_predicate_consuming_test() {
    let mut rng = test_rng();
    let mut context = Context::from_text_rng(
        "<<back1 C . state1 C D\n\
             <<back2 E F . state2 E F\n\
             <<back1 A . <<back2 B A = ()",
        &mut rng,
    )
    .unwrap();

    context.print();

    let replacement_atom = context.core.rules[0].inputs[0][2].atom;
    let replacement = context
        .string_cache
        .atom_to_str(replacement_atom)
        .unwrap()
        .to_string();
    assert!(replacement.starts_with("D_BACK"));

    assert_eq!(
        context.core.rules,
        [RuleBuilder::new(
            vec![
                tokenize(
                    &format!("state1 A {}", replacement),
                    &mut context.string_cache
                ),
                tokenize("state2 B A", &mut context.string_cache)
            ],
            vec![],
            LineColSpan {
                line_start: 3,
                line_end: 3,
                col_start: 1,
                col_end: 30
            }
        )
        .build(0)]
    );
}

#[test]
fn context_from_text_backwards_predicate_update_test() {
    let mut rng = test_rng();
    let mut context = Context::from_text_rng(
        "state1 foo bar . once
             <<back C D . ?state1 C D\n\
             once . <<back A B = foo A B",
        &mut rng,
    )
    .unwrap();

    context.update().unwrap();
    context.print();

    assert_eq!(
        context.core.state.get_all(),
        [
            tokenize("state1 foo bar", &mut context.string_cache),
            tokenize("foo foo bar", &mut context.string_cache),
        ]
    );
}

#[test]
fn context_from_text_backwards_predicate_update2_test() {
    let mut rng = test_rng();
    let mut context = Context::from_text_rng(
        "state1 foo bar
             <<back C D . ?state1 C D\n\
             state1 foo bar . <<back A B = foo A B",
        &mut rng,
    )
    .unwrap();

    context.print();
    context.update().unwrap();
    context.print();

    assert_eq!(
        context.core.state.get_all(),
        [tokenize("foo foo bar", &mut context.string_cache),]
    );
}

#[test]
fn context_from_text_backwards_predicate_constant_test() {
    let mut rng = test_rng();
    let mut context = Context::from_text_rng(
        "<<back C bar . ?state C\n\
             <<back A B . test A . foo B = ()",
        &mut rng,
    )
    .unwrap();

    context.print();

    assert_eq!(
        context.core.rules,
        [RuleBuilder::new(
            vec![
                tokenize("?state A", &mut context.string_cache),
                tokenize("test A", &mut context.string_cache),
                tokenize("foo bar", &mut context.string_cache)
            ],
            vec![],
            LineColSpan {
                line_start: 2,
                line_end: 2,
                col_start: 1,
                col_end: 34
            }
        )
        .build(0)]
    );
}

#[test]
fn context_from_text_backwards_predicate_constant2_test() {
    let mut rng = test_rng();
    let mut context = Context::from_text_rng(
        "<<back C D . ?state C . ?state D\n\
             <<back A bar . test A = ()",
        &mut rng,
    )
    .unwrap();

    context.print();

    assert_eq!(
        context.core.rules,
        [RuleBuilder::new(
            vec![
                tokenize("?state A", &mut context.string_cache),
                tokenize("?state bar", &mut context.string_cache),
                tokenize("test A", &mut context.string_cache),
            ],
            vec![],
            LineColSpan {
                line_start: 2,
                line_end: 2,
                col_start: 1,
                col_end: 28
            }
        )
        .build(0)]
    );
}

#[test]
fn context_from_text_backwards_predicate_permutations_test() {
    let mut context = Context::from_text(
        "<<back1 . ?state11\n\
             <<back1 . ?state12\n\
             <<back2 . ?state21\n\
             <<back2 . ?state22\n\
             <<back1 . <<back2 = ()",
    )
    .unwrap();

    context.print();

    assert_eq!(
        context.core.rules,
        [
            RuleBuilder::new(
                vec![
                    tokenize("?state11", &mut context.string_cache),
                    tokenize("?state21", &mut context.string_cache),
                ],
                vec![],
                LineColSpan {
                    line_start: 5,
                    line_end: 5,
                    col_start: 1,
                    col_end: 24
                }
            )
            .build(0),
            RuleBuilder::new(
                vec![
                    tokenize("?state11", &mut context.string_cache),
                    tokenize("?state22", &mut context.string_cache),
                ],
                vec![],
                LineColSpan {
                    line_start: 5,
                    line_end: 5,
                    col_start: 1,
                    col_end: 24
                }
            )
            .build(1),
            RuleBuilder::new(
                vec![
                    tokenize("?state12", &mut context.string_cache),
                    tokenize("?state21", &mut context.string_cache),
                ],
                vec![],
                LineColSpan {
                    line_start: 5,
                    line_end: 5,
                    col_start: 1,
                    col_end: 24
                }
            )
            .build(2),
            RuleBuilder::new(
                vec![
                    tokenize("?state12", &mut context.string_cache),
                    tokenize("?state22", &mut context.string_cache),
                ],
                vec![],
                LineColSpan {
                    line_start: 5,
                    line_end: 5,
                    col_start: 1,
                    col_end: 24
                }
            )
            .build(3),
        ]
    );
}

#[test]
fn context_from_text_wildcard_test() {
    let mut context = Context::from_text(
        "test1 . _ = any _\n\
             $test2 _ = _ any",
    )
    .unwrap();

    assert_eq!(
        context.core.rules,
        [
            RuleBuilder::new(
                vec![
                    tokenize("test1", &mut context.string_cache),
                    tokenize("WILDCARD0", &mut context.string_cache)
                ],
                vec![tokenize("any WILDCARD1", &mut context.string_cache)],
                LineColSpan {
                    line_start: 1,
                    line_end: 1,
                    col_start: 1,
                    col_end: 18
                }
            )
            .build(0),
            RuleBuilder::new(
                vec![tokenize("test2 WILDCARD2", &mut context.string_cache)],
                vec![
                    tokenize("test2 WILDCARD2", &mut context.string_cache),
                    tokenize("WILDCARD3 any", &mut context.string_cache),
                ],
                LineColSpan {
                    line_start: 2,
                    line_end: 2,
                    col_start: 1,
                    col_end: 17
                }
            )
            .build(1)
        ]
    );
}

#[test]
fn context_from_text_comment_test() {
    let mut context = Context::from_text(
        "// comment 1\n\
             state 1\n\
             /* comment\n\
             2 */",
    )
    .unwrap();

    assert_eq!(
        context.core.state.get_all(),
        [tokenize("state 1", &mut context.string_cache),]
    );
}

#[test]
fn context_from_text_comment_state_test() {
    let mut context = Context::from_text(
        "#test: {\n\
             \n\
             // comment 1\n\
             in1 = out1\n\
             \n\
             // comment 2\n\
             in2 = out2\n\
             }",
    )
    .unwrap();

    context.print();

    assert_eq!(
        context.core.rules,
        [
            RuleBuilder::new(
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("in1", &mut context.string_cache)
                ],
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("out1", &mut context.string_cache),
                ],
                LineColSpan {
                    line_start: 4,
                    line_end: 4,
                    col_start: 1,
                    col_end: 11
                }
            )
            .build(0),
            RuleBuilder::new(
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("in2", &mut context.string_cache)
                ],
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize("out2", &mut context.string_cache),
                ],
                LineColSpan {
                    line_start: 7,
                    line_end: 7,
                    col_start: 1,
                    col_end: 11
                }
            )
            .build(1),
            // automatically added rule to avoid infinite loops
            RuleBuilder::new(
                vec![
                    tokenize("#test", &mut context.string_cache),
                    tokenize(parser::QUI, &mut context.string_cache)
                ],
                vec![],
                LineColSpan {
                    line_start: 1,
                    line_end: 8,
                    col_start: 1,
                    col_end: 2
                }
            )
            .build(2)
        ]
    );
}

#[test]
fn context_append_state_test() {
    let mut context = Context::from_text("test 1 2").unwrap();

    context.push_state("test 3 4");

    assert_eq!(
        context.core.state.get_all(),
        vec![
            tokenize("test 1 2", &mut context.string_cache),
            tokenize("test 3 4", &mut context.string_cache),
        ]
    );
}

#[test]
fn context_find_matching_rules_test() {
    let mut context = Context::from_text(
        "test 1 2 . test 3 4 . test 5 6\n\
             \n\
             test 1 2 . test 5 6 = match\n\
             test 1 2 . nomatch = nomatch\n\
             test 3 4 . test 5 6 = match",
    )
    .unwrap();

    assert_eq!(
        context.find_matching_rules(&mut |_: &Phrase| None).unwrap(),
        [
            RuleBuilder::new(
                vec![
                    tokenize("test 1 2", &mut context.string_cache),
                    tokenize("test 5 6", &mut context.string_cache),
                ],
                vec![tokenize("match", &mut context.string_cache)],
                LineColSpan {
                    line_start: 3,
                    line_end: 3,
                    col_start: 1,
                    col_end: 28
                }
            )
            .build(0),
            RuleBuilder::new(
                vec![
                    tokenize("test 3 4", &mut context.string_cache),
                    tokenize("test 5 6", &mut context.string_cache),
                ],
                vec![tokenize("match", &mut context.string_cache)],
                LineColSpan {
                    line_start: 5,
                    line_end: 5,
                    col_start: 1,
                    col_end: 28
                }
            )
            .build(2),
        ]
    );
}

#[test]
fn update_test() {
    let mut context = Context::from_text(
        "at 0 0 wood . at 0 1 wood . at 1 1 wood . at 0 1 fire . #update\n\
             #update: {\n\
             at X Y wood . at X Y fire = at X Y fire\n\
             () = #spread\n\
             }\n\
             #spread . $at X Y fire . + X 1 X' . + Y' 1 Y = at X' Y fire . at X Y' fire",
    )
    .unwrap()
    .with_test_rng();

    context.print();
    context.update().unwrap();
    context.print();

    assert_eq!(
        context.core.state.get_all(),
        [
            tokenize("at 1 1 wood", &mut context.string_cache),
            tokenize("at 0 0 wood", &mut context.string_cache),
            tokenize("at 0 1 fire", &mut context.string_cache),
            tokenize("at 1 1 fire", &mut context.string_cache),
            tokenize("at 0 0 fire", &mut context.string_cache),
        ]
    );
}

#[test]
fn update2_test() {
    // check that the open/close depths for (bar*) are handled correctly for variable assignment
    let mut context = Context::from_text(
        "test\n\
             test = foo1 (1) . foo2 ((2)) . foo3 (((3)))\n\
             foo1 BAR = BAR\n\
             foo2 (BAR) = BAR\n\
             foo3 (BAR) = BAR",
    )
    .unwrap()
    .with_test_rng();

    context.print();
    context.update().unwrap();
    context.print();

    let mut actual = context
        .core
        .state
        .get_all()
        .iter()
        .map(|p| phrase_to_string(p, &context.string_cache))
        .collect::<Vec<_>>();
    actual.sort();

    assert_eq!(actual, vec!["((3))", "(1)", "(2)"]);
}

#[test]
fn token_test() {
    let mut string_cache = StringCache::new();
    assert!(!is_var_token(&Token::new("tt1", 1, 1, &mut string_cache)));
    assert!(!is_var_token(&Token::new("tT1", 1, 1, &mut string_cache)));
    assert!(!is_var_token(&Token::new("1", 1, 1, &mut string_cache)));
    assert!(!is_var_token(&Token::new("1Tt", 1, 1, &mut string_cache)));
    assert!(!is_var_token(&Token::new("", 1, 1, &mut string_cache)));
    assert!(is_var_token(&Token::new("T", 1, 1, &mut string_cache)));
    assert!(is_var_token(&Token::new("TT1", 1, 1, &mut string_cache)));
    assert!(is_var_token(&Token::new("TT1'", 1, 1, &mut string_cache)));
}

#[test]
fn rule_matches_state_truthiness_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            rule_new(
                vec![
                    tokenize("t1 t3 t2", &mut string_cache),
                    tokenize("t1 t2 t3", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("t1 t3 t2", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(vec![tokenize("t1 T2 T3", &mut string_cache)], vec![]),
            vec![tokenize("t1 t2 t3", &mut string_cache)],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("t1 T3 T2", &mut string_cache),
                    tokenize("t1 T2 T3", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("t1 t2 t3", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("t1 T3 T2", &mut string_cache),
                    tokenize("t1 T2 T3", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("t1 t3 t2", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("t1 T1 T2", &mut string_cache),
                    tokenize("+ T1 T2 T2", &mut string_cache),
                ],
                vec![],
            ),
            vec![tokenize("t1 1 2", &mut string_cache)],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("t1 T1 T2", &mut string_cache),
                    tokenize("+ T1 T2 T2", &mut string_cache),
                ],
                vec![],
            ),
            vec![tokenize("t1 0 2", &mut string_cache)],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("t1 T1 T2", &mut string_cache),
                    tokenize("t1 T1 T2", &mut string_cache),
                ],
                vec![],
            ),
            vec![tokenize("t1 0 2", &mut string_cache)],
            false,
        ),
        // successful match with backwards predicates at first and last position
        (
            rule_new(
                vec![
                    tokenize("+ T1 T2 T2", &mut string_cache),
                    tokenize("t1 T1 T2", &mut string_cache),
                    tokenize("t3 T3 T4", &mut string_cache),
                    tokenize("+ T3 T4 T2", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("t1 0 2", &mut string_cache),
                tokenize("t3 -2 4", &mut string_cache),
            ],
            true,
        ),
        // unsuccessful match with backwards predicates at first and last position
        (
            rule_new(
                vec![
                    tokenize("+ T1 T2 T2", &mut string_cache),
                    tokenize("t1 T1 T2", &mut string_cache),
                    tokenize("t3 T3 T4", &mut string_cache),
                    tokenize("+ T3 T4 0", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("t1 0 2", &mut string_cache),
                tokenize("t3 -2 4", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("+ T1 1 T2", &mut string_cache),
                    tokenize("+ T3 1 T4", &mut string_cache),
                    tokenize("t1 T1 T4", &mut string_cache),
                ],
                vec![],
            ),
            vec![tokenize("t1 0 1", &mut string_cache)],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("first", &mut string_cache),
                    // failing backwards predicate
                    tokenize("+ 3 4 5", &mut string_cache),
                    tokenize("at X Y fire", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("at 1 1 fire", &mut string_cache),
                tokenize("at 1 -1 fire", &mut string_cache),
                tokenize("first", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("at X Y fire", &mut string_cache),
                    tokenize("< 0 X", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("at 0 0 fire", &mut string_cache),
                tokenize("at 2 0 fire", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("has RESULT", &mut string_cache),
                    tokenize("RESULT", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("has foo", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("has RESULT", &mut string_cache),
                    tokenize("RESULT", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("has bar", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(
                vec![tokenize("(nested-first X) Y", &mut string_cache)],
                vec![tokenize("X Y", &mut string_cache)],
            ),
            vec![tokenize("a b", &mut string_cache)],
            false,
        ),
    ];

    for (rule, state, expected) in test_cases.into_iter() {
        let mut state = State::from_phrases(&state);

        let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None).unwrap();

        if expected {
            assert!(result.is_some());
        } else {
            assert!(result.is_none());
        }
    }
}

#[test]
fn rule_matches_state_truthiness_negated_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            rule_new(vec![tokenize("!test", &mut string_cache)], vec![]),
            vec![
                tokenize("foo", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(vec![tokenize("!test", &mut string_cache)], vec![]),
            vec![
                tokenize("foo", &mut string_cache),
                tokenize("test", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("test", &mut string_cache),
                    tokenize("!test", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("foo", &mut string_cache),
                tokenize("test", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("test", &mut string_cache),
                    tokenize("!test", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("foo", &mut string_cache),
                tokenize("test", &mut string_cache),
                tokenize("test", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("!test A B", &mut string_cache),
                    tokenize("foo A B", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("foo 1 2", &mut string_cache),
                tokenize("test 1 3", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("!test A B", &mut string_cache),
                    tokenize("foo A B", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("foo 1 2", &mut string_cache),
                tokenize("test 1 2", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(
                vec![
                    tokenize("has RESULT", &mut string_cache),
                    tokenize("!RESULT", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("has bar", &mut string_cache),
                tokenize("foo", &mut string_cache),
            ],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("has RESULT", &mut string_cache),
                    tokenize("!RESULT", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("has (a b)", &mut string_cache),
                tokenize("a b", &mut string_cache),
            ],
            false,
        ),
    ];

    for (rule, state, expected) in test_cases.into_iter() {
        let mut state = State::from_phrases(&state);

        let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None).unwrap();

        if expected {
            assert!(result.is_some());
        } else {
            assert!(result.is_none());
        }
    }
}

#[test]
fn rule_matches_state_truthiness_nonconsuming_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            rule_new(
                vec![
                    tokenize("?t1 T1 T2", &mut string_cache),
                    tokenize("?t1 T1 T2", &mut string_cache),
                ],
                vec![],
            ),
            vec![tokenize("t1 0 2", &mut string_cache)],
            true,
        ),
        (
            rule_new(
                vec![
                    tokenize("?test", &mut string_cache),
                    tokenize("?!test", &mut string_cache),
                ],
                vec![],
            ),
            vec![
                tokenize("foo", &mut string_cache),
                tokenize("test", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            false,
        ),
        (
            rule_new(vec![tokenize("?!test", &mut string_cache)], vec![]),
            vec![
                tokenize("foo", &mut string_cache),
                tokenize("bar", &mut string_cache),
            ],
            true,
        ),
    ];

    for (rule, state, expected) in test_cases.into_iter() {
        let mut state = State::from_phrases(&state);

        let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None).unwrap();

        if expected {
            assert!(result.is_some());
        } else {
            assert!(result.is_none());
        }
    }
}

#[test]
fn rule_matches_state_output_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            rule_new(
                vec![
                    tokenize("t1 T1 T2", &mut string_cache),
                    tokenize("+ T1 T2 T3'", &mut string_cache),
                    tokenize("+ T1 T4' T2", &mut string_cache),
                ],
                vec![
                    tokenize("t3 T3'", &mut string_cache),
                    tokenize("t4 T4'", &mut string_cache),
                ],
            ),
            vec![tokenize("t1 3 4", &mut string_cache)],
            rule_new(
                vec![tokenize("t1 3 4", &mut string_cache)],
                vec![
                    tokenize("t3 7", &mut string_cache),
                    tokenize("t4 1", &mut string_cache),
                ],
            ),
        ),
        (
            rule_new(
                vec![
                    tokenize("#collision", &mut string_cache),
                    tokenize("block-falling ID X Y", &mut string_cache),
                    tokenize("+ Y1 1 Y", &mut string_cache),
                    tokenize("block-set ID2 X Y1", &mut string_cache),
                ],
                vec![
                    tokenize("block-setting ID X Y", &mut string_cache),
                    tokenize("#collision", &mut string_cache),
                    tokenize("block-set ID2 X Y1", &mut string_cache),
                ],
            ),
            vec![
                tokenize("block-set 0 5 2", &mut string_cache),
                tokenize("block-set 1 6 1", &mut string_cache),
                tokenize("block-set 3 6 0", &mut string_cache),
                tokenize("block-falling 7 6 2", &mut string_cache),
                tokenize("#collision", &mut string_cache),
                tokenize("block-falling 6 5 2", &mut string_cache),
                tokenize("block-set 2 5 0", &mut string_cache),
            ],
            rule_new(
                vec![
                    tokenize("#collision", &mut string_cache),
                    tokenize("block-falling 7 6 2", &mut string_cache),
                    tokenize("block-set 1 6 1", &mut string_cache),
                ],
                vec![
                    tokenize("block-setting 7 6 2", &mut string_cache),
                    tokenize("#collision", &mut string_cache),
                    tokenize("block-set 1 6 1", &mut string_cache),
                ],
            ),
        ),
        (
            rule_new(
                vec![
                    tokenize("foo RESULT", &mut string_cache),
                    tokenize("RESULT", &mut string_cache),
                ],
                vec![
                    tokenize("bar RESULT", &mut string_cache),
                    tokenize("RESULT", &mut string_cache),
                ],
            ),
            vec![
                tokenize("foo 1", &mut string_cache),
                tokenize("1", &mut string_cache),
            ],
            rule_new(
                vec![
                    tokenize("foo 1", &mut string_cache),
                    tokenize("1", &mut string_cache),
                ],
                vec![
                    tokenize("bar 1", &mut string_cache),
                    tokenize("1", &mut string_cache),
                ],
            ),
        ),
        (
            rule_new(
                vec![
                    tokenize("foo RESULT", &mut string_cache),
                    tokenize("RESULT", &mut string_cache),
                ],
                vec![
                    tokenize("bar RESULT", &mut string_cache),
                    tokenize("RESULT", &mut string_cache),
                ],
            ),
            vec![
                tokenize("foo (a (b c))", &mut string_cache),
                tokenize("a (b c)", &mut string_cache),
            ],
            rule_new(
                vec![
                    tokenize("foo (a (b c))", &mut string_cache),
                    tokenize("a (b c)", &mut string_cache),
                ],
                vec![
                    tokenize("bar (a (b c))", &mut string_cache),
                    tokenize("a (b c)", &mut string_cache),
                ],
            ),
        ),
        // rule with one-way modulo backwards predicate surrounded by two-way predicates
        (
            rule_new(
                vec![
                    tokenize("draw X", &mut string_cache),
                    tokenize("- X 7 X'", &mut string_cache),
                    tokenize("+ X' 1 X''", &mut string_cache),
                    tokenize("% X'' 10 X'''", &mut string_cache),
                    tokenize("+ X''' 7 X''''", &mut string_cache),
                ],
                vec![tokenize("draw X''''", &mut string_cache)],
            ),
            vec![tokenize("draw 19", &mut string_cache)],
            rule_new(
                vec![tokenize("draw 19", &mut string_cache)],
                vec![tokenize("draw 10", &mut string_cache)],
            ),
        ),
        // rule with variable as first atom in inputs
        (
            rule_new(
                vec![
                    tokenize("PARENT is parent of CHILD", &mut string_cache),
                    tokenize("AUNT is sister of PARENT", &mut string_cache),
                    tokenize("AUNT is parent of COUSIN", &mut string_cache),
                ],
                vec![tokenize("COUSIN is cousin of CHILD", &mut string_cache)],
            ),
            vec![
                tokenize("Mary is parent of Sarah", &mut string_cache),
                tokenize("David is parent of Tom", &mut string_cache),
                tokenize("Mary is sister of David", &mut string_cache),
            ],
            rule_new(
                vec![
                    tokenize("David is parent of Tom", &mut string_cache),
                    tokenize("Mary is sister of David", &mut string_cache),
                    tokenize("Mary is parent of Sarah", &mut string_cache),
                ],
                vec![tokenize("Sarah is cousin of Tom", &mut string_cache)],
            ),
        ),
    ];

    for (rule, state, expected) in test_cases.into_iter() {
        let mut state = State::from_phrases(&state);

        let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None).unwrap();
        println!("state:\n{}", state::state_to_string(&state, &string_cache));

        assert!(
            result.is_some(),
            "\nrule: {}",
            rule.to_string(&string_cache)
        );

        let actual = result.unwrap().rule;
        assert_eq!(
            actual,
            expected,
            "\nactual: {}\nexpected: {}",
            actual.to_string(&string_cache),
            expected.to_string(&string_cache),
        );
    }
}

#[test]
fn rule_matches_state_input_side_predicate_none_test() {
    let mut string_cache = StringCache::new();

    let rule = rule_new(
        vec![tokenize("^test A", &mut string_cache)],
        vec![tokenize("A", &mut string_cache)],
    );
    let mut state = State::new();

    let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None).unwrap();

    assert!(result.is_none());
}

#[test]
fn rule_matches_state_input_side_predicate_some_fail1_test() {
    let mut string_cache = StringCache::new();

    let rule = rule_new(
        vec![tokenize("^test A", &mut string_cache)],
        vec![tokenize("A", &mut string_cache)],
    );
    let mut state = State::new();

    let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| {
        Some(tokenize("^nah no", &mut string_cache))
    })
    .unwrap();

    assert!(result.is_none());
}

#[test]
fn rule_matches_state_input_side_predicate_some_fail2_test() {
    let mut string_cache = StringCache::new();

    let rule = rule_new(
        vec![
            tokenize("^test A", &mut string_cache),
            tokenize("+ 1 1 A", &mut string_cache),
        ],
        vec![tokenize("A", &mut string_cache)],
    );
    let mut state = State::new();

    let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| {
        Some(tokenize("^test 3", &mut string_cache))
    })
    .unwrap();

    assert!(result.is_none());
}

#[test]
fn rule_matches_state_input_side_predicate_some_pass1_test() {
    let mut string_cache = StringCache::new();

    let rule = rule_new(
        vec![
            tokenize("^test B", &mut string_cache),
            tokenize("+ 2 3 A", &mut string_cache),
            tokenize("% A 4 B", &mut string_cache),
        ],
        vec![tokenize("B", &mut string_cache)],
    );
    let mut state = State::new();

    let result = rule_matches_state(&rule, &mut state, &mut |p: &Phrase| {
        assert_eq!(
            p.get(1).and_then(|t| StringCache::atom_to_integer(t.atom)),
            Some(1)
        );
        Some(vec![])
    })
    .unwrap();

    assert!(result.is_some());
    assert_eq!(
        result.unwrap().rule,
        rule_new(vec![], vec![tokenize("1", &mut string_cache)])
    );
}

#[test]
fn rule_matches_state_input_side_predicate_some_pass2_test() {
    let mut string_cache = StringCache::new();

    let rule = rule_new(
        vec![
            tokenize("^test A", &mut string_cache),
            tokenize("+ 1 A 3", &mut string_cache),
        ],
        vec![tokenize("A", &mut string_cache)],
    );
    let mut state = State::new();

    let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| {
        Some(tokenize("^test 2", &mut string_cache))
    })
    .unwrap();

    assert!(result.is_some());
    assert_eq!(
        result.unwrap().rule,
        rule_new(vec![], vec![tokenize("2", &mut string_cache)])
    );
}

#[test]
fn rule_matches_state_input_side_predicate_some_pass3_test() {
    let mut string_cache = StringCache::new();

    let rule = rule_new(
        vec![
            tokenize("^test A", &mut string_cache),
            tokenize("< 3 A", &mut string_cache),
        ],
        vec![tokenize("A", &mut string_cache)],
    );
    let mut state = State::new();

    let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| {
        Some(tokenize("^test 4", &mut string_cache))
    })
    .unwrap();

    assert!(result.is_some());
    assert_eq!(
        result.unwrap().rule,
        rule_new(vec![], vec![tokenize("4", &mut string_cache)])
    );
}

#[test]
fn evaluate_backwards_pred_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            tokenize("+ A 2 3", &mut string_cache),
            Some(tokenize("+ 1 2 3", &mut string_cache)),
        ),
        (
            tokenize("+ 1 B 3", &mut string_cache),
            Some(tokenize("+ 1 2 3", &mut string_cache)),
        ),
        (
            tokenize("+ 1 2 C", &mut string_cache),
            Some(tokenize("+ 1 2 3", &mut string_cache)),
        ),
        (tokenize("+ 1 2 4", &mut string_cache), None),
        (
            tokenize("!== (a b c) (a b)", &mut string_cache),
            Some(tokenize("!== (a b c) (a b)", &mut string_cache)),
        ),
        (tokenize("!== (a b c) (a b c)", &mut string_cache), None),
        (
            tokenize("== (a b c) (a b c)", &mut string_cache),
            Some(tokenize("== (a b c) (a b c)", &mut string_cache)),
        ),
        (
            tokenize("% 13 3 M", &mut string_cache),
            Some(tokenize("% 13 3 1", &mut string_cache)),
        ),
        (
            tokenize("% 14 3 2", &mut string_cache),
            Some(tokenize("% 14 3 2", &mut string_cache)),
        ),
    ];

    for (input, expected) in test_cases.into_iter() {
        assert_eq!(evaluate_backwards_pred(&input), expected);
    }
}

#[test]
fn assign_state_vars_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            tokenize("+ T1 T2 T3", &mut string_cache),
            vec![tokenize("1 2 3", &mut string_cache)],
            vec![
                MatchLite {
                    var_atom: string_cache.str_to_atom("T1"),
                    var_open_close_depth: (0, 0),
                    var_open_close_depth_norm: (0, 0),
                    state_i: 0,
                    state_token_range: (0, 1),
                },
                MatchLite {
                    var_atom: string_cache.str_to_atom("T2"),
                    var_open_close_depth: (0, 0),
                    var_open_close_depth_norm: (0, 0),
                    state_i: 0,
                    state_token_range: (1, 2),
                },
            ],
            tokenize("+ 1 2 T3", &mut string_cache),
        ),
        (
            tokenize("T1 (T2 T3)", &mut string_cache),
            vec![
                tokenize("t11 t12", &mut string_cache),
                tokenize("t31 (t32 t33)", &mut string_cache),
            ],
            vec![
                MatchLite {
                    var_atom: string_cache.str_to_atom("T1"),
                    var_open_close_depth: (1, 0),
                    var_open_close_depth_norm: (0, 0),
                    state_i: 0,
                    state_token_range: (0, 2),
                },
                MatchLite {
                    var_atom: string_cache.str_to_atom("T3"),
                    var_open_close_depth: (0, 2),
                    var_open_close_depth_norm: (0, 1),
                    state_i: 1,
                    state_token_range: (0, 3),
                },
            ],
            tokenize("(t11 t12) (T2 (t31 (t32 t33)))", &mut string_cache),
        ),
        (
            tokenize("T1 !T2", &mut string_cache),
            vec![tokenize("t11 t12", &mut string_cache)],
            vec![MatchLite {
                var_atom: string_cache.str_to_atom("T2"),
                var_open_close_depth: (0, 1),
                var_open_close_depth_norm: (0, 0),
                state_i: 0,
                state_token_range: (0, 2),
            }],
            tokenize("T1 (!t11 t12)", &mut string_cache),
        ),
    ];

    for (tokens, state, matches, expected) in test_cases.into_iter() {
        let state = State::from_phrases(&state);
        let actual = assign_state_vars(&tokens, &state, &matches, &mut PhraseGroupCounter::new());
        assert_eq!(
            actual,
            expected,
            "\ntokens = {}\nactual = {}\nexpected = {}",
            tokens.to_string(&string_cache),
            actual.to_string(&string_cache),
            expected.to_string(&string_cache),
        );
    }
}

#[test]
fn match_variables_test() {
    let mut string_cache = StringCache::new();

    let test_cases = vec![
        (
            tokenize("t1 T2 T3", &mut string_cache),
            tokenize("t1 t2 t3", &mut string_cache),
            Some(vec![
                (
                    string_cache.str_to_atom("T2"),
                    vec![Token::new("t2", 0, 0, &mut string_cache)],
                ),
                (
                    string_cache.str_to_atom("T3"),
                    vec![Token::new("t3", 0, 0, &mut string_cache)],
                ),
            ]),
        ),
        (
            tokenize("t1 T2", &mut string_cache),
            tokenize("t1 (t21 t22)", &mut string_cache),
            Some(vec![(
                string_cache.str_to_atom("T2"),
                tokenize("t21 t22", &mut string_cache),
            )]),
        ),
        (
            tokenize("t1 (t21 T22 t23) T3", &mut string_cache),
            tokenize("t1 (t21 (t221 t222 t223) t23) t3", &mut string_cache),
            Some(vec![
                (
                    string_cache.str_to_atom("T22"),
                    tokenize("t221 t222 t223", &mut string_cache),
                ),
                (
                    string_cache.str_to_atom("T3"),
                    vec![Token::new("t3", 0, 0, &mut string_cache)],
                ),
            ]),
        ),
        (
            tokenize("t1 T2 T3", &mut string_cache),
            tokenize("t1 t2 (t3 t2)", &mut string_cache),
            Some(vec![
                (
                    string_cache.str_to_atom("T2"),
                    vec![Token::new("t2", 0, 0, &mut string_cache)],
                ),
                (
                    string_cache.str_to_atom("T3"),
                    tokenize("t3 t2", &mut string_cache),
                ),
            ]),
        ),
        (
            tokenize("t1 T2", &mut string_cache),
            tokenize("t1 (t2 t3 (t3 t2))", &mut string_cache),
            Some(vec![(
                string_cache.str_to_atom("T2"),
                tokenize("t2 t3 (t3 t2)", &mut string_cache),
            )]),
        ),
        (
            tokenize("t1 T2", &mut string_cache),
            tokenize("t1 ((t2 t3) t3 t2)", &mut string_cache),
            Some(vec![(
                string_cache.str_to_atom("T2"),
                tokenize("(t2 t3) t3 t2", &mut string_cache),
            )]),
        ),
        (
            tokenize("t1 (t2 t3 T2)", &mut string_cache),
            tokenize("t1 (t2 t3 (t3 t2))", &mut string_cache),
            Some(vec![(
                string_cache.str_to_atom("T2"),
                tokenize("t3 t2", &mut string_cache),
            )]),
        ),
        (
            tokenize("t1 (T2 t3 t2)", &mut string_cache),
            tokenize("t1 ((t2 t3) t3 t2)", &mut string_cache),
            Some(vec![(
                string_cache.str_to_atom("T2"),
                tokenize("t2 t3", &mut string_cache),
            )]),
        ),
        (
            tokenize("t1 T2 T3", &mut string_cache),
            tokenize("t1 (t2 t3) (t3 t2)", &mut string_cache),
            Some(vec![
                (
                    string_cache.str_to_atom("T2"),
                    tokenize("t2 t3", &mut string_cache),
                ),
                (
                    string_cache.str_to_atom("T3"),
                    tokenize("t3 t2", &mut string_cache),
                ),
            ]),
        ),
        (
            tokenize("t1 t3", &mut string_cache),
            tokenize("t1 t3", &mut string_cache),
            Some(vec![]),
        ),
        (
            tokenize("t1 t3", &mut string_cache),
            tokenize("t1 (t21 t23)", &mut string_cache),
            None,
        ),
        (
            tokenize("t1 T3", &mut string_cache),
            tokenize("t1 t2 t3", &mut string_cache),
            None,
        ),
        (
            tokenize("t1 T3 t3", &mut string_cache),
            tokenize("t1 t2", &mut string_cache),
            None,
        ),
        (
            tokenize("t1 T3 T3", &mut string_cache),
            tokenize("t1 t2 t3", &mut string_cache),
            None,
        ),
        (
            tokenize("t1 T3 T3", &mut string_cache),
            tokenize("t1 t3 t3", &mut string_cache),
            Some(vec![(
                string_cache.str_to_atom("T3"),
                vec![Token::new("t3", 0, 0, &mut string_cache)],
            )]),
        ),
    ];

    for (input_tokens, pred_tokens, expected) in test_cases.into_iter() {
        let actual = match_variables_with_existing(&input_tokens, &pred_tokens);
        assert_eq!(
            actual,
            expected,
            "\ninput = {}\npred = {}",
            input_tokens.to_string(&string_cache),
            pred_tokens.to_string(&string_cache)
        );
    }
}

#[test]
fn match_variables_with_existing_test() {
    let mut string_cache = StringCache::new();

    let input_tokens = tokenize("T1 T2 T3", &mut string_cache);
    let state = State::from_phrases(&[tokenize("t1 t2 t3", &mut string_cache)]);

    let mut matches = vec![MatchLite {
        var_atom: string_cache.str_to_atom("T2"),
        var_open_close_depth: (0, 0),
        var_open_close_depth_norm: (0, 0),
        state_i: 0,
        state_token_range: (1, 2),
    }];

    let result = match_state_variables_with_existing(&input_tokens, &state, 0, &mut matches);

    assert!(result);

    assert_eq!(
        matches,
        [
            MatchLite {
                var_atom: string_cache.str_to_atom("T2"),
                var_open_close_depth: (0, 0),
                var_open_close_depth_norm: (0, 0),
                state_i: 0,
                state_token_range: (1, 2),
            },
            MatchLite {
                var_atom: string_cache.str_to_atom("T1"),
                var_open_close_depth: (1, 0),
                var_open_close_depth_norm: (0, 0),
                state_i: 0,
                state_token_range: (0, 1),
            },
            MatchLite {
                var_atom: string_cache.str_to_atom("T3"),
                var_open_close_depth: (0, 1),
                var_open_close_depth_norm: (0, 0),
                state_i: 0,
                state_token_range: (2, 3),
            },
        ]
    )
}

#[test]
fn test_extend_state_from_context() {
    let mut context1 = Context::from_text("foo 1\nbar 1").unwrap();
    let context2 = Context::from_text("foo 2\nbar 2").unwrap();
    context1.extend_state_from_context(&context2);

    assert_eq!(
        context1.core.state.get_all(),
        [
            tokenize("foo 1", &mut context1.string_cache),
            tokenize("bar 1", &mut context1.string_cache),
            tokenize("foo 2", &mut context1.string_cache),
            tokenize("bar 2", &mut context1.string_cache),
        ]
    );
}

#[test]
fn tokenize_test() {
    let mut string_cache = StringCache::new();

    assert_eq!(
        tokenize("t1", &mut string_cache),
        [Token::new("t1", 1, 1, &mut string_cache)]
    );

    assert_ne!(
        tokenize("?t1", &mut string_cache),
        [Token::new("t1", 1, 1, &mut string_cache)]
    );

    assert_eq!(
        tokenize("t1 (t21 (t221 t222 t223) t23) t3", &mut string_cache),
        [
            Token::new("t1", 1, 0, &mut string_cache),
            Token::new("t21", 1, 0, &mut string_cache),
            Token::new("t221", 1, 0, &mut string_cache),
            Token::new("t222", 0, 0, &mut string_cache),
            Token::new("t223", 0, 1, &mut string_cache),
            Token::new("t23", 0, 1, &mut string_cache),
            Token::new("t3", 0, 1, &mut string_cache),
        ]
    );

    assert_eq!(
        tokenize("t1 t2 (((t3 )) t4)", &mut string_cache),
        [
            Token::new("t1", 1, 0, &mut string_cache),
            Token::new("t2", 0, 0, &mut string_cache),
            Token::new("t3", 3, 2, &mut string_cache),
            Token::new("t4", 0, 2, &mut string_cache),
        ]
    );

    assert_eq!(
        tokenize("(t1 t2) (t3 t4)", &mut string_cache),
        [
            Token::new("t1", 2, 0, &mut string_cache),
            Token::new("t2", 0, 1, &mut string_cache),
            Token::new("t3", 1, 0, &mut string_cache),
            Token::new("t4", 0, 2, &mut string_cache),
        ]
    );

    assert_eq!(
        tokenize("t1 t2 (t3'' t4')", &mut string_cache),
        [
            Token::new("t1", 1, 0, &mut string_cache),
            Token::new("t2", 0, 0, &mut string_cache),
            Token::new("t3''", 1, 0, &mut string_cache),
            Token::new("t4'", 0, 2, &mut string_cache),
        ]
    );
}

#[test]
fn tokenize_var_test() {
    let mut string_cache = StringCache::new();

    assert_eq!(
        tokenize("A", &mut string_cache),
        [Token::new("A", 0, 0, &mut string_cache)]
    );

    assert_eq!(
        tokenize("(A)", &mut string_cache),
        [Token::new("A", 1, 1, &mut string_cache)]
    );
}

#[test]
fn tokenize_string_test() {
    let mut string_cache = StringCache::new();

    // parse quoted variables as strings
    assert_eq!(
        tokenize("\"X\"", &mut string_cache),
        [Token {
            atom: string_cache.str_to_atom("X"),
            flag: TokenFlag::None,
            is_negated: false,
            is_consuming: true,
            open_depth: 1,
            close_depth: 1,
        }]
    );

    assert_eq!(
        tokenize("\"string here\"", &mut string_cache),
        [Token::new("string here", 1, 1, &mut string_cache),]
    );

    assert_eq!(
        tokenize("\"one string\" \"two strings\"", &mut string_cache),
        [
            Token::new("one string", 1, 0, &mut string_cache),
            Token::new("two strings", 0, 1, &mut string_cache),
        ]
    );

    assert_eq!(
        tokenize(
            "t1 t2 (((\"string here\" )) \"final string\")",
            &mut string_cache
        ),
        [
            Token::new("t1", 1, 0, &mut string_cache),
            Token::new("t2", 0, 0, &mut string_cache),
            Token::new("string here", 3, 2, &mut string_cache),
            Token::new("final string", 0, 2, &mut string_cache),
        ]
    );
}

#[test]
fn tokenize_wildcard_test() {
    {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("_", &mut string_cache),
            [Token::new("WILDCARD0", 0, 0, &mut string_cache),]
        );
    }

    {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("t1 t2 (((_ )) _)", &mut string_cache),
            [
                Token::new("t1", 1, 0, &mut string_cache),
                Token::new("t2", 0, 0, &mut string_cache),
                Token::new("WILDCARD0", 3, 2, &mut string_cache),
                Token::new("WILDCARD1", 0, 2, &mut string_cache),
            ]
        );
    }

    {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("_ _test _TEST", &mut string_cache),
            [
                Token::new("WILDCARD0", 1, 0, &mut string_cache),
                Token::new("_test", 0, 0, &mut string_cache),
                Token::new("WILDCARD1", 0, 1, &mut string_cache),
            ]
        );
    }
}

#[test]
fn phrase_groups_simple_test() {
    let mut string_cache = StringCache::new();
    let phrase = tokenize("1 2 3", &mut string_cache);

    assert_eq!(
        phrase.groups().collect::<Vec<_>>(),
        vec![
            &[Token::new_integer(1, 1, 0)],
            &[Token::new_integer(2, 0, 0)],
            &[Token::new_integer(3, 0, 1)]
        ]
    );
}

#[test]
fn phrase_groups_single_test() {
    let mut string_cache = StringCache::new();
    let phrase = tokenize("1", &mut string_cache);

    assert_eq!(
        phrase.groups().collect::<Vec<_>>(),
        vec![&[Token::new_integer(1, 1, 1)],]
    );
}

#[test]
fn phrase_groups_compound_test() {
    let mut string_cache = StringCache::new();
    let phrase = tokenize("((1 2) 3) (4 (5 6)", &mut string_cache);

    assert_eq!(
        phrase.groups().collect::<Vec<_>>(),
        vec![
            &[
                Token::new_integer(1, 3, 0),
                Token::new_integer(2, 0, 1),
                Token::new_integer(3, 0, 1)
            ],
            &[
                Token::new_integer(4, 1, 0),
                Token::new_integer(5, 1, 0),
                Token::new_integer(6, 0, 2)
            ]
        ]
    );

    assert_eq!(
        phrase.groups_at_depth(2).collect::<Vec<_>>(),
        vec![
            &[Token::new_integer(1, 3, 0), Token::new_integer(2, 0, 1)][..],
            &[Token::new_integer(3, 0, 1)][..],
            &[Token::new_integer(4, 1, 0)][..],
            &[Token::new_integer(5, 1, 0), Token::new_integer(6, 0, 2)][..]
        ]
    );

    assert_eq!(
        phrase.groups_at_depth(3).collect::<Vec<_>>(),
        vec![
            &[Token::new_integer(1, 3, 0)],
            &[Token::new_integer(2, 0, 1)],
            &[Token::new_integer(5, 1, 0)],
            &[Token::new_integer(6, 0, 2)]
        ]
    );

    assert_eq!(
        phrase
            .groups()
            .nth(0)
            .unwrap()
            .groups_at_depth(2)
            .collect::<Vec<_>>(),
        vec![
            &[Token::new_integer(1, 3, 0), Token::new_integer(2, 0, 1)][..],
            &[Token::new_integer(3, 0, 1)][..],
        ]
    );
}

#[test]
fn phrase_normalize_compound1_test() {
    let mut string_cache = StringCache::new();
    let original_phrase = tokenize("((1 2) 3) (4 (5 6))", &mut string_cache);

    let mut phrase = original_phrase.clone();
    let len = phrase.len();
    phrase[0].open_depth += 1;
    phrase[len - 1].close_depth += 3;

    assert_eq!(phrase.normalize(), original_phrase);
}

#[test]
fn phrase_normalize_compound2_test() {
    let mut string_cache = StringCache::new();
    let original_phrase = tokenize("((1 2) 3) (4 (5 6))", &mut string_cache);

    let mut phrase = original_phrase.clone();
    let len = phrase.len();
    phrase[0].open_depth += 3;
    phrase[len - 1].close_depth -= 1;

    assert_eq!(phrase.normalize(), original_phrase);
}

#[test]
fn state_roundtrip_test() {
    let mut string_cache = StringCache::new();

    let mut state = State::new();
    state.push(tokenize("test 123", &mut string_cache));

    assert_eq!(state.get_all(), [tokenize("test 123", &mut string_cache)]);
}

#[test]
fn test_match_without_variables_test() {
    let mut string_cache = StringCache::new();

    // test complicated match that caused integer overflow in the past
    let input_tokens = tokenize("ui-action ID RESULT HINT", &mut string_cache);
    let pred_tokens = tokenize("ui-action character-recruit (on-tick 1 2 (character-recruit (3 4))) (Recruit Logan \"to your team\")", &mut string_cache);

    let result = test_match_without_variables(&input_tokens, &pred_tokens);
    assert!(result.is_some());
}

#[test]
fn test_match_without_variables2_test() {
    let mut string_cache = StringCache::new();

    // test complicated match that caused integer overflow in the past
    let mut input_tokens = tokenize("RESULT", &mut string_cache);
    input_tokens[0].open_depth = 0;
    input_tokens[0].close_depth = 0;

    let pred_tokens = tokenize("ui-action character-recruit (on-tick 1 2 (character-recruit (3 4))) (Recruit Logan \"to your team\")", &mut string_cache);

    let result = test_match_without_variables(&input_tokens, &pred_tokens);
    assert!(result.is_some());
}

#[test]
fn update_rule_repeat_error_test() {
    let mut context = Context::from_text(
        "foo\n\
             foo = foo",
    )
    .unwrap();

    let result = context.update();
    assert!(matches!(
        result,
        Err(update::Error::RuleRepeatError(update::RuleRepeatError {
            rule_id: 0
        }))
    ));

    if let Err(e) = result {
        let rule = e.rule(&context.core.rules).unwrap();
        assert_eq!(
            rule.source_span,
            LineColSpan {
                line_start: 2,
                line_end: 2,
                col_start: 1,
                col_end: 10
            }
        );
    }
}

#[test]
fn find_phrases_test() {
    let mut context =
        Context::from_text("foo . a foo . a b bar . a foo c . c foo b . bar b c . a foo b c")
            .unwrap();
    let foo_atom = context.str_to_atom("foo");
    let phrases: Vec<VecPhrase> = context
        .core
        .state
        .iter_pattern([None, Some(foo_atom), None], false)
        .map(|phrase_id| context.core.state.get(phrase_id).to_vec())
        .collect();

    for phrase in &phrases {
        println!("{}", phrase_to_string(&phrase, &context.string_cache));
    }
    assert_eq!(
        phrases,
        [
            tokenize("a foo c", &mut context.string_cache),
            tokenize("c foo b", &mut context.string_cache),
            tokenize("a foo b c", &mut context.string_cache),
        ]
    );
}

#[test]
fn find_phrases_exact_length_test() {
    let mut context =
        Context::from_text("foo . a foo . a b bar . a foo c . c foo b . bar b c . a foo b c")
            .unwrap();
    let foo_atom = context.str_to_atom("foo");
    let phrases: Vec<VecPhrase> = context
        .core
        .state
        .iter_pattern([None, Some(foo_atom), None], true)
        .map(|phrase_id| context.core.state.get(phrase_id).to_vec())
        .collect();
    assert_eq!(
        phrases,
        [
            tokenize("a foo c", &mut context.string_cache),
            tokenize("c foo b", &mut context.string_cache),
        ]
    );
}

#[test]
fn remove_pattern_test() {
    let mut context =
        Context::from_text("foo . a foo . a b bar . a foo c . c foo b . bar b c . a foo b c")
            .unwrap();
    let foo_atom = context.str_to_atom("foo");
    context
        .core
        .state
        .remove_pattern([None, Some(foo_atom), None], false);
    assert_eq!(
        context.core.state.get_all(),
        [
            tokenize("foo", &mut context.string_cache),
            tokenize("a foo", &mut context.string_cache),
            tokenize("a b bar", &mut context.string_cache),
            tokenize("bar b c", &mut context.string_cache),
        ]
    );
}

#[test]
fn remove_phrase_exact_length_test() {
    let mut context =
        Context::from_text("foo . a foo . a b bar . a foo c . c foo b . bar b c . a foo b c")
            .unwrap();
    let foo_atom = context.str_to_atom("foo");
    context
        .core
        .state
        .remove_pattern([None, Some(foo_atom), None], true);
    assert_eq!(
        context.core.state.get_all(),
        [
            tokenize("foo", &mut context.string_cache),
            tokenize("a foo", &mut context.string_cache),
            tokenize("a b bar", &mut context.string_cache),
            tokenize("bar b c", &mut context.string_cache),
            tokenize("a foo b c", &mut context.string_cache),
        ]
    );
}

fn rule_new(inputs: Vec<Vec<Token>>, outputs: Vec<Vec<Token>>) -> Rule {
    RuleBuilder::new(
        inputs,
        outputs,
        LineColSpan {
            line_start: 0,
            line_end: 0,
            col_start: 0,
            col_end: 0,
        },
    )
    .build(0)
}

fn match_variables_with_existing(
    input_tokens: &Phrase,
    pred_tokens: &Phrase,
) -> Option<Vec<(Atom, Vec<Token>)>> {
    let mut result = vec![];

    let mut state = State::new();
    state.push(pred_tokens.to_vec());

    if match_state_variables_with_existing(input_tokens, &state, 0, &mut result) {
        Some(
            result
                .iter()
                .map(|m| (m.var_atom, m.to_phrase(&state)))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    }
}

pub(crate) fn test_rng() -> SmallRng {
    SmallRng::seed_from_u64(123)
}
