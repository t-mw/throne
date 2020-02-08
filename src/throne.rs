use rand;
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};
use scratchpad::{
    uninitialized_boxed_slice_for_bytes, uninitialized_boxed_slice_for_markers, CacheAligned,
    Scratchpad,
};

use std::fmt;
use std::mem::MaybeUninit;
use std::vec::Vec;

use crate::matching::*;
use crate::parser;
use crate::rule::Rule;
use crate::state::State;
use crate::string_cache::{Atom, StringCache};
use crate::token::*;

#[allow(unused_macros)]
macro_rules! dump {
    ($($a:expr),*) => ({
        let mut txt = format!("{}:{}:", file!(), line!());
        $({txt += &format!("\t{}={:?};", stringify!($a), $a)});*;
        println!("DEBUG: {}", txt);
    })
}

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Vec<Token>> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Vec<Token>> {}

#[derive(Clone)]
pub struct Context {
    pub core: Core,
    pub string_cache: StringCache,
}

#[derive(Clone)]
pub struct Core {
    pub state: State,
    pub rules: Vec<Rule>,
    rng: SmallRng,
    qui_atom: Atom,
}

impl Core {
    pub fn rule_matches_state<F>(&self, rule: &Rule, mut side_input: F) -> bool
    where
        F: SideInput,
    {
        const POOL_SIZE: usize = 1usize * 1024 * 1024; // 1 MB
        const MARKERS_MAX: usize = 16;

        let pool =
            unsafe { uninitialized_boxed_slice_for_bytes::<MaybeUninit<CacheAligned>>(POOL_SIZE) };
        let tracking = unsafe {
            uninitialized_boxed_slice_for_markers::<MaybeUninit<CacheAligned>>(MARKERS_MAX)
        };

        let mut scratchpad = Scratchpad::new(pool, tracking);

        rule_matches_state(
            rule,
            &mut self.state.clone(),
            &mut side_input,
            &mut scratchpad,
        )
        .is_some()
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
}

pub struct ContextBuilder<'a> {
    text: &'a str,
    string_cache: StringCache,
    rng: Option<&'a mut SmallRng>,
}

impl<'a> ContextBuilder<'a> {
    pub fn new() -> Self {
        ContextBuilder {
            text: "",
            string_cache: StringCache::new(),
            rng: None,
        }
    }

    pub fn text(mut self, text: &'a str) -> Self {
        self.text = text;
        self
    }

    pub fn string_cache(mut self, string_cache: StringCache) -> Self {
        self.string_cache = string_cache;
        self
    }

    pub fn rng(mut self, rng: &'a mut SmallRng) -> Self {
        self.rng = Some(rng);
        self
    }

    pub fn build(self) -> Context {
        let mut default_rng = SmallRng::from_rng(&mut thread_rng()).unwrap();
        Context::new(
            self.text,
            self.string_cache,
            self.rng.unwrap_or(&mut default_rng),
        )
    }
}

impl Context {
    pub fn from_text(text: &str) -> Self {
        ContextBuilder::new().text(text).build()
    }

    #[cfg(test)]
    pub(crate) fn from_text_rng(text: &str, rng: &mut SmallRng) -> Self {
        ContextBuilder::new().text(text).rng(rng).build()
    }

    fn new(text: &str, mut string_cache: StringCache, rng: &mut SmallRng) -> Context {
        let mut result = parser::parse(text, &mut string_cache, rng);

        let mut state = State::new();
        for phrase in result.state.drain(..) {
            state.push(phrase);
        }

        let seed = [
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
        ];

        let rng = SmallRng::from_seed(seed);

        state.update_first_atoms();
        let qui_atom = string_cache.str_to_atom("qui");

        Context {
            core: Core {
                state,
                rules: result.rules,
                rng,
                qui_atom,
            },
            string_cache,
        }
    }

    pub fn extend_state_from_context(&mut self, other: &Context) {
        for phrase_id in other.core.state.iter() {
            let phrase = other.core.state.get(phrase_id);
            let new_phrase = phrase
                .iter()
                .map(|t| {
                    if StringCache::atom_to_number(t.atom).is_some() {
                        t.clone()
                    } else {
                        let string = other
                            .string_cache
                            .atom_to_str(t.atom)
                            .expect(&format!("missing token: {:?}", t));

                        let mut new_token = t.clone();
                        new_token.atom = self.string_cache.str_to_atom(string);
                        new_token
                    }
                })
                .collect();
            self.core.state.push(new_phrase);
        }

        self.core.state.update_first_atoms();
    }

    pub fn str_to_atom(&mut self, text: &str) -> Atom {
        self.string_cache.str_to_atom(text)
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.string_cache.str_to_existing_atom(text)
    }

    pub fn atom_to_str(&self, atom: Atom) -> &str {
        self.string_cache.atom_to_str(atom).unwrap()
    }

    pub fn atom_to_number(&self, atom: Atom) -> Option<i32> {
        StringCache::atom_to_number(atom)
    }

    pub fn with_test_rng(mut self) -> Context {
        self.core.rng = test_rng();

        self
    }

    pub fn append_state(&mut self, text: &str) {
        self.core.state.push(tokenize(text, &mut self.string_cache));
    }

    pub fn find_matching_rules<F>(&self, mut side_input: F) -> Vec<Rule>
    where
        F: SideInput,
    {
        let state = &mut self.core.state.clone();

        const POOL_SIZE: usize = 1usize * 1024 * 1024; // 1 MB
        const MARKERS_MAX: usize = 16;

        let pool =
            unsafe { uninitialized_boxed_slice_for_bytes::<MaybeUninit<CacheAligned>>(POOL_SIZE) };
        let tracking = unsafe {
            uninitialized_boxed_slice_for_markers::<MaybeUninit<CacheAligned>>(MARKERS_MAX)
        };

        let mut scratchpad = Scratchpad::new(pool, tracking);

        self.core
            .rules
            .iter()
            .filter_map(|rule| rule_matches_state(&rule, state, &mut side_input, &mut scratchpad))
            .collect()
    }

    pub fn update<F>(&mut self, side_input: F)
    where
        F: SideInput,
    {
        update(&mut self.core, side_input);
    }

    pub fn execute_rule(&mut self, rule: &Rule) {
        execute_rule(rule, &mut self.core.state);
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn find_phrase<'a>(&'a self, a1: Option<&Atom>) -> Option<&'a [Token]> {
        self.find_phrase2(a1, None)
    }

    pub fn find_phrase2<'a>(&'a self, a1: Option<&Atom>, a2: Option<&Atom>) -> Option<&'a [Token]> {
        self.find_phrase3(a1, a2, None)
    }

    pub fn find_phrase3<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
    ) -> Option<&'a [Token]> {
        self.find_phrase4(a1, a2, a3, None)
    }

    pub fn find_phrase4<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
        a4: Option<&Atom>,
    ) -> Option<&'a [Token]> {
        self.find_phrase5(a1, a2, a3, a4, None)
    }

    pub fn find_phrase5<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
        a4: Option<&Atom>,
        a5: Option<&Atom>,
    ) -> Option<&'a [Token]> {
        for phrase_id in self.core.state.iter() {
            let p = self.core.state.get(phrase_id);

            match (
                p.get(0).map(|t| &t.atom),
                p.get(1).map(|t| &t.atom),
                p.get(2).map(|t| &t.atom),
                p.get(3).map(|t| &t.atom),
                p.get(4).map(|t| &t.atom),
            ) {
                (s1, s2, s3, s4, s5)
                    if (a1.is_none() || a1 == s1)
                        && (a2.is_none() || a2 == s2)
                        && (a3.is_none() || a3 == s3)
                        && (a4.is_none() || a4 == s4)
                        && (a5.is_none() || a5 == s5) =>
                {
                    return Some(p);
                }
                _ => (),
            };
        }

        None
    }

    pub fn find_phrases<'a>(&'a self, a1: Option<&Atom>) -> Vec<&'a [Token]> {
        self.find_phrases2(a1, None)
    }

    pub fn find_phrases2<'a>(&'a self, a1: Option<&Atom>, a2: Option<&Atom>) -> Vec<&'a [Token]> {
        self.find_phrases3(a1, a2, None)
    }

    pub fn find_phrases3<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.find_phrases4(a1, a2, a3, None)
    }

    pub fn find_phrases4<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
        a4: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.find_phrases5(a1, a2, a3, a4, None)
    }

    pub fn find_phrases5<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
        a4: Option<&Atom>,
        a5: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.core
            .state
            .iter()
            .filter_map(|phrase_id| {
                let p = self.core.state.get(phrase_id);

                if match (
                    p.get(0).map(|t| &t.atom),
                    p.get(1).map(|t| &t.atom),
                    p.get(2).map(|t| &t.atom),
                    p.get(3).map(|t| &t.atom),
                    p.get(4).map(|t| &t.atom),
                ) {
                    (s1, s2, s3, s4, s5) => {
                        (a1.is_none() || a1 == s1)
                            && (a2.is_none() || a2 == s2)
                            && (a3.is_none() || a3 == s3)
                            && (a4.is_none() || a4 == s4)
                            && (a5.is_none() || a5 == s5)
                    }
                } {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    // --- find phrases with with exact length ---

    pub fn find_phrases_exactly1<'a>(&'a self, a1: Option<&Atom>) -> Vec<&'a [Token]> {
        self.core
            .state
            .iter()
            .filter_map(|phrase_id| {
                let p = self.core.state.get(phrase_id);

                if p.len() == 1 && (a1.is_none() || a1 == Some(&p[0].atom)) {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn find_phrases_exactly2<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.core
            .state
            .iter()
            .filter_map(|phrase_id| {
                let p = self.core.state.get(phrase_id);

                if p.len() == 2
                    && (a1.is_none() || a1 == Some(&p[0].atom))
                    && (a2.is_none() || a2 == Some(&p[1].atom))
                {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn find_phrases_exactly3<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.core
            .state
            .iter()
            .filter_map(|phrase_id| {
                let p = self.core.state.get(phrase_id);

                if p.len() == 3
                    && (a1.is_none() || a1 == Some(&p[0].atom))
                    && (a2.is_none() || a2 == Some(&p[1].atom))
                    && (a3.is_none() || a3 == Some(&p[2].atom))
                {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn find_phrases_exactly4<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
        a4: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.core
            .state
            .iter()
            .filter_map(|phrase_id| {
                let p = self.core.state.get(phrase_id);

                if p.len() == 4
                    && (a1.is_none() || a1 == Some(&p[0].atom))
                    && (a2.is_none() || a2 == Some(&p[1].atom))
                    && (a3.is_none() || a3 == Some(&p[2].atom))
                    && (a4.is_none() || a4 == Some(&p[3].atom))
                {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn find_phrases_exactly5<'a>(
        &'a self,
        a1: Option<&Atom>,
        a2: Option<&Atom>,
        a3: Option<&Atom>,
        a4: Option<&Atom>,
        a5: Option<&Atom>,
    ) -> Vec<&'a [Token]> {
        self.core
            .state
            .iter()
            .filter_map(|phrase_id| {
                let p = self.core.state.get(phrase_id);

                if p.len() == 5
                    && (a1.is_none() || a1 == Some(&p[0].atom))
                    && (a2.is_none() || a2 == Some(&p[1].atom))
                    && (a3.is_none() || a3 == Some(&p[2].atom))
                    && (a4.is_none() || a4 == Some(&p[3].atom))
                    && (a5.is_none() || a5 == Some(&p[4].atom))
                {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let state = build_state(&self.core.state, &self.string_cache);

        let mut rules = self
            .core
            .rules
            .iter()
            .map(|r| r.to_string(&self.string_cache))
            .collect::<Vec<_>>();
        rules.sort();

        write!(f, "state:\n{}\nrules:\n{}", state, rules.join("\n"))
    }
}

fn execute_rule(rule: &Rule, state: &mut State) {
    let inputs = &rule.inputs;
    let outputs = &rule.outputs;

    inputs.iter().for_each(|input| {
        if input[0].is_consuming {
            state.remove_phrase(input);
        }
    });

    outputs.iter().for_each(|output| {
        state.push(output.clone());
    });
}

pub fn update<F>(core: &mut Core, mut side_input: F)
where
    F: SideInput,
{
    let state = &mut core.state;
    let rules = &mut core.rules;
    let rng = &mut core.rng;

    const POOL_SIZE: usize = 1usize * 1024 * 1024; // 1 MB
    const MARKERS_MAX: usize = 16;

    let pool =
        unsafe { uninitialized_boxed_slice_for_bytes::<MaybeUninit<CacheAligned>>(POOL_SIZE) };
    let tracking =
        unsafe { uninitialized_boxed_slice_for_markers::<MaybeUninit<CacheAligned>>(MARKERS_MAX) };

    let mut scratchpad = Scratchpad::new(pool, tracking);

    // shuffle state so that a given rule with multiple potential
    // matches does not always match the same permutation of state.
    state.shuffle(rng);

    // shuffle rules so that each has an equal chance of selection.
    rng.shuffle(rules);

    // change starting rule on each iteration to introduce randomness.
    let mut start_rule_idx = 0;

    let mut quiescence = false;

    loop {
        let mut matching_rule = None;

        if quiescence {
            state.push(vec![Token::new_atom(core.qui_atom, 0, 0)]);
        }

        state.update_first_atoms();

        for i in 0..rules.len() {
            let rule = &rules[(start_rule_idx + i) % rules.len()];

            if let Some(rule) = rule_matches_state(&rule, state, &mut side_input, &mut scratchpad) {
                matching_rule = Some(rule);
                break;
            }
        }

        start_rule_idx += 1;

        if quiescence {
            quiescence = false;

            if matching_rule.is_none() {
                let qui_atom = core.qui_atom;
                assert!(
                    state
                        .iter()
                        .enumerate()
                        .filter(|&(_, p)| state.get(p)[0].atom == qui_atom)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                        == vec![state.len() - 1],
                    "expected 1 * () at end of state"
                );

                let idx = state.len() - 1;
                state.remove_idx(idx);

                return;
            }
        }

        if let Some(ref matching_rule) = matching_rule {
            execute_rule(matching_rule, state);
        } else {
            quiescence = true;
        }
    }
}

pub trait PhraseString {
    fn to_string(&self, string_cache: &StringCache) -> String;
}

impl PhraseString for Phrase {
    fn to_string(&self, string_cache: &StringCache) -> String {
        build_phrase(self, string_cache)
    }
}

pub fn build_state(state: &State, string_cache: &StringCache) -> String {
    state
        .iter()
        .map(|phrase_id| build_phrase(state.get(phrase_id), string_cache))
        .collect::<Vec<_>>()
        .join("\n")
}

fn test_rng() -> SmallRng {
    let seed = [
        123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123,
    ];

    SmallRng::from_seed(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule_new(inputs: Vec<Vec<Token>>, outputs: Vec<Vec<Token>>) -> Rule {
        Rule::new(0, inputs, outputs)
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
                    .map(|m| (m.atom, m.to_phrase(&state)))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        }
    }

    #[test]
    fn context_from_text_state_test() {
        let mut context = Context::from_text(
            "at 0 0 wood . at 0 0 wood . #update \n\
             at 1 2 wood",
        );

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
    fn context_from_text_rules_test() {
        let mut context = Context::from_text(
            "at 0 0 wood . at 1 2 wood = at 1 0 wood \n\
             asdf . #test: { \n\
             at 3 4 wood = at 5 6 wood . at 7 8 wood \n\
             }",
        );

        assert_eq!(
            context.core.rules,
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("at 0 0 wood", &mut context.string_cache),
                        tokenize("at 1 2 wood", &mut context.string_cache),
                    ],
                    vec![tokenize("at 1 0 wood", &mut context.string_cache)]
                ),
                Rule::new(
                    1,
                    vec![
                        tokenize("#test", &mut context.string_cache),
                        tokenize("at 3 4 wood", &mut context.string_cache),
                        tokenize("asdf", &mut context.string_cache),
                    ],
                    vec![
                        tokenize("at 5 6 wood", &mut context.string_cache),
                        tokenize("at 7 8 wood", &mut context.string_cache),
                        tokenize("#test", &mut context.string_cache),
                    ],
                ),
            ]
        );
    }

    #[test]
    fn context_from_text_copy_test() {
        let mut context = Context::from_text("at 0 0 wood . $at 1 2 wood = at 1 0 wood");

        assert_eq!(
            context.core.rules,
            [Rule::new(
                0,
                vec![
                    tokenize("at 0 0 wood", &mut context.string_cache),
                    tokenize("at 1 2 wood", &mut context.string_cache),
                ],
                vec![
                    tokenize("at 1 2 wood", &mut context.string_cache),
                    tokenize("at 1 0 wood", &mut context.string_cache)
                ]
            ),]
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
             = `broken\ntext`",
        );

        assert_eq!(
            context.core.rules,
            [
                Rule::new(
                    0,
                    vec![tokenize("broken line 1", &mut context.string_cache)],
                    vec![
                        tokenize("broken line 2", &mut context.string_cache),
                        tokenize("broken line 3", &mut context.string_cache),
                        tokenize("broken line 4", &mut context.string_cache),
                    ],
                ),
                Rule::new(
                    1,
                    vec![tokenize("text", &mut context.string_cache)],
                    vec![tokenize("`broken\ntext`", &mut context.string_cache)],
                )
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
        );
        // intentionally separate backwards predicates by simple 'test' atom to test parsing

        context.print();

        assert_eq!(
            context.core.rules,
            [Rule::new(
                0,
                vec![
                    tokenize(
                        "?state1 A D_BACK36822967802071690107",
                        &mut context.string_cache
                    ),
                    tokenize("test", &mut context.string_cache),
                    tokenize("?state2 B A", &mut context.string_cache)
                ],
                vec![]
            )]
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
        );

        context.print();

        assert_eq!(
            context.core.rules,
            [Rule::new(
                0,
                vec![
                    tokenize(
                        "state1 A D_BACK36822967802071690107",
                        &mut context.string_cache
                    ),
                    tokenize("state2 B A", &mut context.string_cache)
                ],
                vec![]
            )]
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
        );

        context.update(|_: &Phrase| None);

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
        );

        context.print();
        context.update(|_: &Phrase| None);

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
        );

        context.print();

        assert_eq!(
            context.core.rules,
            [Rule::new(
                0,
                vec![
                    tokenize("?state A", &mut context.string_cache),
                    tokenize("test A", &mut context.string_cache),
                    tokenize("foo bar", &mut context.string_cache)
                ],
                vec![]
            )]
        );
    }

    #[test]
    fn context_from_text_backwards_predicate_constant2_test() {
        let mut rng = test_rng();
        let mut context = Context::from_text_rng(
            "<<back C D . ?state C . ?state D\n\
             <<back A bar . test A = ()",
            &mut rng,
        );

        context.print();

        assert_eq!(
            context.core.rules,
            [Rule::new(
                0,
                vec![
                    tokenize("?state A", &mut context.string_cache),
                    tokenize("?state bar", &mut context.string_cache),
                    tokenize("test A", &mut context.string_cache),
                ],
                vec![]
            )]
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
        );

        context.print();

        assert_eq!(
            context.core.rules,
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("?state11", &mut context.string_cache),
                        tokenize("?state21", &mut context.string_cache),
                    ],
                    vec![]
                ),
                Rule::new(
                    1,
                    vec![
                        tokenize("?state11", &mut context.string_cache),
                        tokenize("?state22", &mut context.string_cache),
                    ],
                    vec![]
                ),
                Rule::new(
                    2,
                    vec![
                        tokenize("?state12", &mut context.string_cache),
                        tokenize("?state21", &mut context.string_cache),
                    ],
                    vec![]
                ),
                Rule::new(
                    3,
                    vec![
                        tokenize("?state12", &mut context.string_cache),
                        tokenize("?state22", &mut context.string_cache),
                    ],
                    vec![]
                ),
            ]
        );
    }

    #[test]
    fn context_from_text_wildcard_test() {
        let mut context = Context::from_text(
            "test1 . _ = any _\n\
             $test2 _ = _ any",
        );

        assert_eq!(
            context.core.rules,
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("test1", &mut context.string_cache),
                        tokenize("WILDCARD0", &mut context.string_cache)
                    ],
                    vec![tokenize("any WILDCARD1", &mut context.string_cache)]
                ),
                Rule::new(
                    1,
                    vec![tokenize("test2 WILDCARD2", &mut context.string_cache)],
                    vec![
                        tokenize("test2 WILDCARD2", &mut context.string_cache),
                        tokenize("WILDCARD3 any", &mut context.string_cache),
                    ]
                )
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
        );

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
        );

        assert_eq!(
            context.core.rules,
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("#test", &mut context.string_cache),
                        tokenize("in1", &mut context.string_cache)
                    ],
                    vec![
                        tokenize("out1", &mut context.string_cache),
                        tokenize("#test", &mut context.string_cache)
                    ],
                ),
                Rule::new(
                    1,
                    vec![
                        tokenize("#test", &mut context.string_cache),
                        tokenize("in2", &mut context.string_cache)
                    ],
                    vec![
                        tokenize("out2", &mut context.string_cache),
                        tokenize("#test", &mut context.string_cache)
                    ],
                )
            ]
        );
    }

    #[test]
    fn context_append_state_test() {
        let mut context = Context::from_text("test 1 2");

        context.append_state("test 3 4");

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
        );

        assert_eq!(
            context.find_matching_rules(&mut |_: &Phrase| None),
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("test 1 2", &mut context.string_cache),
                        tokenize("test 5 6", &mut context.string_cache),
                    ],
                    vec![tokenize("match", &mut context.string_cache)]
                ),
                Rule::new(
                    2,
                    vec![
                        tokenize("test 3 4", &mut context.string_cache),
                        tokenize("test 5 6", &mut context.string_cache),
                    ],
                    vec![tokenize("match", &mut context.string_cache)],
                ),
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
        .with_test_rng();

        context.update(|_: &Phrase| None);

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

        let mut test_cases = vec![
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
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let mut state = State::from_phrases(&state);

            let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None);

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

        let mut test_cases = vec![
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
                    tokenize("has bar", &mut string_cache),
                    tokenize("bar", &mut string_cache),
                ],
                false,
            ),
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let mut state = State::from_phrases(&state);

            let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None);

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

        let mut test_cases = vec![
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

        for (rule, state, expected) in test_cases.drain(..) {
            let mut state = State::from_phrases(&state);

            let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None);

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

        let mut test_cases = vec![
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
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let mut state = State::from_phrases(&state);

            let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None);

            assert!(result.is_some());

            let actual = result.unwrap();
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

        let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None);

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
        });

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
        });

        assert!(result.is_none());
    }

    #[test]
    fn rule_matches_state_input_side_predicate_some_pass1_test() {
        let mut string_cache = StringCache::new();

        let rule = rule_new(
            vec![
                tokenize("^test A", &mut string_cache),
                tokenize("+ 2 3 A", &mut string_cache),
            ],
            vec![tokenize("A", &mut string_cache)],
        );
        let mut state = State::new();

        let result = rule_matches_state(&rule, &mut state, &mut |p: &Phrase| {
            assert_eq!(
                p.get(1).and_then(|t| StringCache::atom_to_number(t.atom)),
                Some(5)
            );
            Some(vec![])
        });

        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
            rule_new(vec![], vec![tokenize("5", &mut string_cache)])
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
        });

        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
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
        });

        assert!(result.is_some());
        assert_eq!(
            result.unwrap(),
            rule_new(vec![], vec![tokenize("4", &mut string_cache)])
        );
    }

    #[test]
    fn evaluate_backwards_pred_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
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
        ];

        for (input, expected) in test_cases.drain(..) {
            assert_eq!(evaluate_backwards_pred(&input), expected);
        }
    }

    #[test]
    fn assign_state_vars_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                tokenize("+ T1 T2 T3", &mut string_cache),
                vec![tokenize("1 2 3", &mut string_cache)],
                vec![
                    MatchLite {
                        atom: string_cache.str_to_atom("T1"),
                        state_i: 0,
                        depths: (0, 0),
                        range: (0, 1),
                    },
                    MatchLite {
                        atom: string_cache.str_to_atom("T2"),
                        state_i: 0,
                        depths: (0, 0),
                        range: (1, 2),
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
                        atom: string_cache.str_to_atom("T1"),
                        state_i: 0,
                        depths: (0, 0),
                        range: (0, 2),
                    },
                    MatchLite {
                        atom: string_cache.str_to_atom("T3"),
                        state_i: 1,
                        depths: (0, 0),
                        range: (0, 3),
                    },
                ],
                tokenize("(t11 t12) (T2 (t31 (t32 t33)))", &mut string_cache),
            ),
            (
                tokenize("T1 !T2", &mut string_cache),
                vec![tokenize("t11 t12", &mut string_cache)],
                vec![MatchLite {
                    atom: string_cache.str_to_atom("T2"),
                    state_i: 0,
                    depths: (0, 0),
                    range: (0, 2),
                }],
                tokenize("T1 (!t11 t12)", &mut string_cache),
            ),
        ];

        for (tokens, state, matches, expected) in test_cases.drain(..) {
            let state = State::from_phrases(&state);
            assert_eq!(assign_state_vars(&tokens, &state, &matches), expected);
        }
    }

    #[test]
    fn match_variables_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                tokenize("t1 T2 T3", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
                Some(vec![
                    (
                        string_cache.str_to_atom("T2"),
                        tokenize("t2", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T3"),
                        tokenize("t3", &mut string_cache),
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
                        tokenize("t3", &mut string_cache),
                    ),
                ]),
            ),
            (
                tokenize("t1 T2 T3", &mut string_cache),
                tokenize("t1 t2 (t3 t2)", &mut string_cache),
                Some(vec![
                    (
                        string_cache.str_to_atom("T2"),
                        tokenize("t2", &mut string_cache),
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
                    tokenize("t3", &mut string_cache),
                )]),
            ),
        ];

        for (input_tokens, pred_tokens, expected) in test_cases.drain(..) {
            assert_eq!(
                match_variables_with_existing(&input_tokens, &pred_tokens),
                expected
            );
        }
    }

    #[test]
    fn match_variables_with_existing_test() {
        let mut string_cache = StringCache::new();

        let input_tokens = tokenize("T1 T2 T3", &mut string_cache);
        let state = State::from_phrases(&[tokenize("t1 t2 t3", &mut string_cache)]);

        let mut matches = vec![MatchLite {
            atom: string_cache.str_to_atom("T2"),
            state_i: 0,
            depths: (0, 0),
            range: (1, 2),
        }];

        let result = match_state_variables_with_existing(&input_tokens, &state, 0, &mut matches);

        assert!(result);

        assert_eq!(
            matches,
            [
                MatchLite {
                    atom: string_cache.str_to_atom("T2"),
                    state_i: 0,
                    depths: (0, 0),
                    range: (1, 2),
                },
                MatchLite {
                    atom: string_cache.str_to_atom("T1"),
                    state_i: 0,
                    depths: (1, 0),
                    range: (0, 1),
                },
                MatchLite {
                    atom: string_cache.str_to_atom("T3"),
                    state_i: 0,
                    depths: (0, 1),
                    range: (2, 3),
                },
            ]
        )
    }

    #[test]
    fn test_extend_state_from_context() {
        let mut context1 = Context::from_text("foo 1\nbar 1");
        let context2 = Context::from_text("foo 2\nbar 2");
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
            [Token::new("t1", 0, 0, &mut string_cache)]
        );

        assert_ne!(
            tokenize("?t1", &mut string_cache),
            [Token::new("t1", 0, 0, &mut string_cache)]
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
                Token::new("t3", 1, 0, &mut string_cache),
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
    fn tokenize_string_test() {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("`string here`", &mut string_cache),
            [Token::new("string here", 0, 0, &mut string_cache),]
        );

        assert_eq!(
            tokenize("`one string` `two strings`", &mut string_cache),
            [
                Token::new("one string", 1, 0, &mut string_cache),
                Token::new("two strings", 0, 1, &mut string_cache),
            ]
        );

        assert_eq!(
            tokenize(
                "t1 t2 (((`string here` )) `final string`)",
                &mut string_cache
            ),
            [
                Token::new("t1", 1, 0, &mut string_cache),
                Token::new("t2", 0, 0, &mut string_cache),
                Token::new("string here", 1, 0, &mut string_cache),
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
                    Token::new("WILDCARD0", 1, 0, &mut string_cache),
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
    fn phrase_get_group_simple_test() {
        let mut string_cache = StringCache::new();
        let phrase = tokenize("1 2 3", &mut string_cache);

        assert_eq!(
            phrase.get_group(0),
            Some(vec![Token::new_number(1, 1, 0)].as_slice())
        );
        assert_eq!(
            phrase.get_group(1),
            Some(vec![Token::new_number(2, 0, 0)].as_slice())
        );
        assert_eq!(
            phrase.get_group(2),
            Some(vec![Token::new_number(3, 0, 1)].as_slice())
        );
    }

    #[test]
    fn phrase_get_group_compound_test() {
        let mut string_cache = StringCache::new();
        let phrase = tokenize("((1 2) 3) (4 (5 6)", &mut string_cache);

        assert_eq!(
            phrase.get_group(0),
            Some(
                vec![
                    Token::new_number(1, 3, 0),
                    Token::new_number(2, 0, 1),
                    Token::new_number(3, 0, 1)
                ]
                .as_slice()
            )
        );

        assert_eq!(
            phrase.get_group(1),
            Some(
                vec![
                    Token::new_number(4, 1, 0),
                    Token::new_number(5, 1, 0),
                    Token::new_number(6, 0, 2)
                ]
                .as_slice()
            )
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
        let pred_tokens = tokenize("ui-action character-recruit (on-tick 1 2 (character-recruit (3 4))) (Recruit Logan `to your team`)", &mut string_cache);

        let result = test_match_without_variables(&input_tokens, &pred_tokens);
        assert!(result.is_some());
    }

    #[test]
    fn test_match_without_variables2_test() {
        let mut string_cache = StringCache::new();

        // test complicated match that caused integer overflow in the past
        let input_tokens = tokenize("RESULT", &mut string_cache);
        let pred_tokens = tokenize("ui-action character-recruit (on-tick 1 2 (character-recruit (3 4))) (Recruit Logan `to your team`)", &mut string_cache);

        let result = test_match_without_variables(&input_tokens, &pred_tokens);
        assert!(result.is_some());
    }
}
