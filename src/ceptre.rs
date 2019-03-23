use rand;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use regex::Regex;

use std::collections::HashMap;
use std::fmt;
use std::i32;
use std::vec::Vec;

macro_rules! dump {
    ($($a:expr),*) => ({
        let mut txt = format!("{}:{}:", file!(), line!());
        $({txt += &format!("\t{}={:?};", stringify!($a), $a)});*;
        println!("DEBUG: {}", txt);
    })
}

mod parser {
    #[derive(Parser)]
    #[grammar = "ceptre.pest"]
    pub struct Parser;
}

pub trait OptionFilter<T> {
    fn option_filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Self;
}

// copy of nightly-only method Option::filter
impl<T> OptionFilter<T> for Option<T> {
    fn option_filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Self {
        if let Some(x) = self {
            if predicate(&x) {
                return Some(x);
            }
        }
        None
    }
}

type AtomIdx = i32;
type RuleId = i32;

const MAX_NUMBER: i32 = 99999;
const MAX_STRING_IDX: AtomIdx = i32::MAX - MAX_NUMBER * 2 - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Atom {
    idx: AtomIdx,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BackwardsPred {
    Plus,
    Lt,
    Gt,
    Lte,
    Gte,
    ModNeg,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TokenFlag {
    None,
    Variable,
    Side,
    BackwardsPred(BackwardsPred),
}

#[derive(Clone, Debug)]
pub struct Token {
    pub string: Atom,
    is_negated: bool,
    flag: TokenFlag,
    open_depth: u8,
    close_depth: u8,
}

impl PartialEq for Token {
    fn eq(&self, other: &Token) -> bool {
        token_equal(self, other, false, None, None)
    }
}
impl Eq for Token {}

impl Token {
    pub fn new(
        string: &str,
        open_depth: u8,
        close_depth: u8,
        string_cache: &mut StringCache,
    ) -> Token {
        let mut string = string;

        let mut is_negated = false;
        let mut is_side = false;
        match string.chars().next().expect("first_char") {
            '!' => {
                is_negated = true;
                string = string.get(1..).expect("get");
            }
            '^' => {
                is_side = true;
                string = string.get(1..).expect("get");
            }
            _ => {}
        }

        let mut chars = string.chars();
        let first_char = chars.next();
        let is_var = first_char.expect("first_char").is_ascii_uppercase()
            && chars.all(|c| c.is_numeric() || !c.is_ascii_lowercase());

        let backwards_pred = match string {
            "+" => Some(BackwardsPred::Plus),
            "<" => Some(BackwardsPred::Lt),
            ">" => Some(BackwardsPred::Gt),
            "<=" => Some(BackwardsPred::Lte),
            ">=" => Some(BackwardsPred::Gte),
            "%%" => Some(BackwardsPred::ModNeg),

            _ => None,
        };

        let atom = string_cache.str_to_atom(string);

        let flag = match (is_var, is_side, backwards_pred) {
            (false, false, None) => TokenFlag::None,
            (true, false, None) => TokenFlag::Variable,
            (false, true, None) => TokenFlag::Side,
            (false, false, Some(v)) => TokenFlag::BackwardsPred(v),
            _ => unreachable!(),
        };

        Token {
            string: atom,
            flag,
            is_negated,
            open_depth,
            close_depth,
        }
    }

    pub fn new_atom(atom: Atom, open_depth: u8, close_depth: u8) -> Token {
        Token {
            string: atom,
            flag: TokenFlag::None,
            is_negated: false,
            open_depth,
            close_depth,
        }
    }

    pub fn new_number(n: i32, open_depth: u8, close_depth: u8) -> Token {
        Token {
            string: StringCache::number_to_atom(n),
            flag: TokenFlag::None,
            is_negated: false,
            open_depth,
            close_depth,
        }
    }

    pub fn as_str<'a>(&self, string_cache: &'a StringCache) -> Option<&'a str> {
        string_cache.atom_to_str(self.string)
    }

    pub fn as_number(&self) -> Option<i32> {
        StringCache::atom_to_number(self.string)
    }
}

pub type Phrase = [Token];

pub trait PhraseGroup {
    fn get_group(&self, n: usize) -> Option<&[Token]>;
}

impl PhraseGroup for Phrase {
    fn get_group(&self, n: usize) -> Option<&[Token]> {
        let mut current_group = 0;
        let mut depth = 0;

        let mut start = None;

        for (i, t) in self.iter().enumerate() {
            let depth_before = depth;

            depth += t.open_depth;
            depth -= t.close_depth;

            if depth_before <= 1 {
                start = Some(i);
            }

            if depth <= 1 {
                if current_group == n {
                    return Some(&self[start.unwrap()..i + 1]);
                }

                current_group += 1;
            }
        }

        None
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

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Vec<Token>> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Vec<Token>> {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    pub id: RuleId,
    pub inputs: Vec<Vec<Token>>,
    pub outputs: Vec<Vec<Token>>,
}

impl Rule {
    pub fn new(id: i32, inputs: Vec<Vec<Token>>, outputs: Vec<Vec<Token>>) -> Rule {
        Rule {
            id,
            inputs,
            outputs,
        }
    }

    pub fn to_string(&self, string_cache: &StringCache) -> String {
        rule_to_string(self, string_cache)
    }
}

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
        rule_matches_state(rule, &mut self.state.clone(), &mut side_input).is_some()
    }
}

#[derive(Clone, Debug)]
pub struct State {
    phrases: Vec<PhraseId>,
    phrase_ranges: Vec<(usize, usize)>,
    tokens: Vec<Token>,
    first_atoms: Vec<(usize, Atom)>,
    scratch_idx: Option<(usize, usize)>,
}

fn extract_first_atoms_rule_input(phrase: &Phrase) -> Option<Atom> {
    if is_concrete_pred(phrase) {
        phrase
            .get(0)
            .option_filter(|t| !is_var_token(t))
            .map(|t| t.string)
    } else {
        None
    }
}

fn extract_first_atoms_state(state: &State) -> Vec<(usize, Atom)> {
    let mut atoms: Vec<(usize, Atom)> = state
        .iter()
        .enumerate()
        .map(|(s_i, phrase_id)| (s_i, state.get(*phrase_id)[0].string))
        .collect();

    atoms.sort_unstable_by(|a, b| a.1.cmp(&b.1));

    atoms
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
}

impl State {
    pub fn new() -> State {
        State {
            phrases: vec![],
            phrase_ranges: vec![],
            tokens: vec![],
            first_atoms: vec![],
            scratch_idx: None,
        }
    }

    fn remove(&mut self, idx: usize) {
        assert!(self.scratch_idx.is_none());

        let remove_id = self.phrases.swap_remove(idx);
        let remove_range = self.phrase_ranges[remove_id.idx];
        let remove_len = remove_range.1 - remove_range.0;

        // after swap_remove, this is the id that will take the old one's place
        let replace_id = PhraseId {
            idx: self.phrase_ranges.len() - 1,
        };
        self.phrase_ranges.swap_remove(remove_id.idx);

        // update the references to the swapped id
        for id in self.phrases.iter_mut() {
            if *id == replace_id {
                *id = remove_id;
            }
        }

        self.tokens.drain(remove_range.0..remove_range.1);

        for range in self.phrase_ranges.iter_mut() {
            if range.0 >= remove_range.1 {
                range.0 -= remove_len;
                range.1 -= remove_len;
            }
        }
    }

    fn update_first_atoms(&mut self) {
        self.first_atoms = extract_first_atoms_state(self);
    }

    fn remove_phrase(&mut self, phrase: &Phrase) {
        let remove_idx = self
            .phrases
            .iter()
            .position(|v| self.get(*v) == phrase)
            .expect("remove_idx");

        self.remove(remove_idx);
    }

    fn shuffle(&mut self, rng: &mut SmallRng) {
        assert!(self.scratch_idx.is_none());
        rng.shuffle(&mut self.phrases);
    }

    pub fn push(&mut self, mut phrase: Vec<Token>) -> PhraseId {
        let begin = self.tokens.len();
        self.tokens.append(&mut phrase);
        let end = self.tokens.len();

        let idx = self.phrase_ranges.len();
        self.phrase_ranges.push((begin, end));

        let id = PhraseId { idx };
        self.phrases.push(id);

        id
    }

    fn len(&self) -> usize {
        self.phrases.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a PhraseId> {
        self.phrases.iter()
    }

    pub fn get(&self, id: PhraseId) -> &Phrase {
        let (begin, end) = self.phrase_ranges[id.idx];
        &self.tokens[begin..end]
    }

    fn get_all(&self) -> Vec<Vec<Token>> {
        self.phrases
            .iter()
            .map(|id| {
                let (begin, end) = self.phrase_ranges[id.idx];
                self.tokens[begin..end].to_vec()
            })
            .collect::<Vec<_>>()
    }

    fn from_phrases(phrases: &[Vec<Token>]) -> State {
        let mut state = State::new();
        for p in phrases {
            state.push(p.clone());
        }
        state.update_first_atoms();
        state
    }

    fn lock_scratch(&mut self) {
        assert!(self.phrases.len() == self.phrase_ranges.len());
        self.scratch_idx = Some((self.phrases.len(), self.tokens.len()));
    }

    fn unlock_scratch(&mut self) {
        self.reset_scratch();
        self.scratch_idx = None;
    }

    fn reset_scratch(&mut self) {
        let (phrase_len, token_len) = self.scratch_idx.expect("scratch_idx");
        self.phrases.drain(phrase_len..);
        self.phrase_ranges.drain(phrase_len..);
        self.tokens.drain(token_len..);
    }
}

impl std::ops::Index<usize> for State {
    type Output = [Token];

    fn index(&self, i: usize) -> &Phrase {
        let id = self.phrases[i];
        self.get(id)
    }
}

#[derive(Clone, Default)]
pub struct StringCache {
    atom_to_str: Vec<String>,
    str_to_atom: HashMap<String, Atom>,
}

impl StringCache {
    pub fn new() -> StringCache {
        Default::default()
    }

    pub fn str_to_atom(&mut self, text: &str) -> Atom {
        use std::str::FromStr;

        if let Some(n) = i32::from_str(text).ok() {
            return StringCache::number_to_atom(n);
        }

        if let Some(atom) = self.str_to_existing_atom(text) {
            return atom;
        }

        let idx = self.atom_to_str.len() as AtomIdx;
        if idx > MAX_STRING_IDX {
            panic!("String cache full");
        }

        let atom = Atom { idx };

        self.atom_to_str.push(text.to_string());
        self.str_to_atom.insert(text.to_string(), atom);

        atom
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.str_to_atom.get(text).cloned()
    }

    fn number_to_atom(n: i32) -> Atom {
        if n.abs() > MAX_NUMBER {
            panic!("{} is large than the maximum of {}", n.abs(), MAX_NUMBER);
        }

        return Atom {
            idx: (n + MAX_NUMBER + 1) + MAX_STRING_IDX,
        };
    }

    pub fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        if atom.idx <= MAX_STRING_IDX {
            Some(&self.atom_to_str[atom.idx as usize])
        } else {
            None
        }
    }

    fn atom_to_number(atom: Atom) -> Option<i32> {
        if atom.idx <= MAX_STRING_IDX {
            None
        } else {
            Some((atom.idx - MAX_STRING_IDX) - MAX_NUMBER - 1)
        }
    }
}

impl Context {
    pub fn from_text(text: &str) -> Context {
        use pest::iterators::Pair;
        use pest::Parser;

        let text = text.replace("()", "qui");

        let file = parser::Parser::parse(parser::Rule::file, &text)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();

        let mut state = State::new();
        let mut rules: Vec<Rule> = vec![];

        let mut string_cache = StringCache::new();

        let mut idx = -1;

        let mut pair_to_ceptre_rule = |pair: Pair<parser::Rule>, string_cache: &mut StringCache| {
            let mut pairs = pair.into_inner();
            let inputs_pair = pairs.next().unwrap();
            let outputs_pair = pairs.next().unwrap();

            idx += 1;

            let mut inputs = vec![];
            let mut outputs = vec![];
            let mut has_input_qui = false;

            for p in inputs_pair.into_inner() {
                let input_phrase = p.into_inner().next().unwrap();

                match input_phrase.as_rule() {
                    parser::Rule::copy_phrase => {
                        let copy_phrase = tokenize(
                            input_phrase.into_inner().next().unwrap().as_str(),
                            string_cache,
                        );

                        inputs.push(copy_phrase.clone());
                        outputs.push(copy_phrase);
                    }
                    rule_type => {
                        has_input_qui = has_input_qui || rule_type == parser::Rule::qui;
                        inputs.push(tokenize(input_phrase.as_str(), string_cache));
                    }
                }
            }

            for p in outputs_pair.into_inner() {
                let output_phrase = p.into_inner().next().unwrap();

                match output_phrase.as_rule() {
                    parser::Rule::qui => (),
                    _ => outputs.push(tokenize(output_phrase.as_str(), string_cache)),
                }
            }

            (Rule::new(idx, inputs, outputs), has_input_qui)
        };

        for line in file.into_inner() {
            match line.as_rule() {
                parser::Rule::stage => {
                    let mut stage = line.into_inner();

                    let prefix_inputs_pair = stage.next().unwrap();
                    let input_phrase_strings = prefix_inputs_pair
                        .into_inner()
                        .map(|p| p.as_str())
                        .collect::<Vec<_>>();

                    for pair in stage {
                        let (mut r, has_input_qui) = pair_to_ceptre_rule(pair, &mut string_cache);

                        let input_phrases = input_phrase_strings
                            .iter()
                            .map(|p| tokenize(p, &mut string_cache))
                            .collect::<Vec<_>>();
                        let stage_phrase_idx = input_phrase_strings
                            .iter()
                            .position(|s| s.chars().next() == Some('#'));

                        // insert stage at beginning, so that 'first atoms' optimization is effective
                        for (i, p) in input_phrases.iter().enumerate() {
                            if stage_phrase_idx == Some(i) {
                                continue;
                            }

                            r.inputs.push(p.clone())
                        }

                        if let Some(i) = stage_phrase_idx {
                            r.inputs.insert(0, input_phrases[i].clone());
                        }

                        if !has_input_qui && stage_phrase_idx.is_some() {
                            r.outputs
                                .push(input_phrases[stage_phrase_idx.unwrap()].clone());
                        }

                        rules.push(r);
                    }
                }
                parser::Rule::rule => {
                    rules.push(pair_to_ceptre_rule(line, &mut string_cache).0);
                }
                parser::Rule::state => {
                    for phrase in line.into_inner() {
                        state.push(tokenize(phrase.as_str(), &mut string_cache));
                    }
                }
                parser::Rule::EOI => (),
                _ => unreachable!("{}", line),
            }
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
                rules,
                rng,
                qui_atom,
            },
            string_cache,
        }
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

        self.core
            .rules
            .iter()
            .filter_map(|rule| rule_matches_state(&rule, state, &mut side_input))
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
            let p = self.core.state.get(*phrase_id);

            match (
                p.get(0).map(|t| &t.string),
                p.get(1).map(|t| &t.string),
                p.get(2).map(|t| &t.string),
                p.get(3).map(|t| &t.string),
                p.get(4).map(|t| &t.string),
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
                let p = self.core.state.get(*phrase_id);

                if match (
                    p.get(0).map(|t| &t.string),
                    p.get(1).map(|t| &t.string),
                    p.get(2).map(|t| &t.string),
                    p.get(3).map(|t| &t.string),
                    p.get(4).map(|t| &t.string),
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
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let state = build_state(&self.core.state, &self.string_cache);

        let mut rules = self
            .core
            .rules
            .iter()
            .map(|r| rule_to_string(r, &self.string_cache))
            .collect::<Vec<_>>();
        rules.sort();

        write!(f, "state:\n{}\nrules:\n{}", state, rules.join("\n"))
    }
}

fn execute_rule(rule: &Rule, state: &mut State) {
    let inputs = &rule.inputs;
    let outputs = &rule.outputs;

    inputs.iter().for_each(|input| {
        state.remove_phrase(input);
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

            if let Some(rule) = rule_matches_state(&rule, state, &mut side_input) {
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
                        .filter(|&(_, p)| state.get(*p)[0].string == qui_atom)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                        == vec![state.len() - 1],
                    "expected 1 * () at end of state"
                );

                let idx = state.len() - 1;
                state.remove(idx);

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

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
fn rule_matches_state<F>(r: &Rule, state: &mut State, side_input: &mut F) -> Option<Rule>
where
    F: SideInput,
{
    let inputs = &r.inputs;
    let outputs = &r.outputs;

    // per input, a list of states that match the input.
    let mut input_state_matches = vec![];

    for (i_i, input) in inputs.iter().enumerate() {
        // TODO: exit early if we already know that side predicate won't match
        if is_concrete_pred(input) {
            let rule_first_atoms = extract_first_atoms_rule_input(input);

            let start_idx = if let Some(first) = rule_first_atoms {
                if let Ok(idx) = state
                    .first_atoms
                    .binary_search_by(|probe| probe.1.cmp(&first))
                {
                    // binary search won't always find the first match,
                    // so search backwards until we find it
                    state
                        .first_atoms
                        .iter()
                        .enumerate()
                        .rev()
                        .skip(state.first_atoms.len() - 1 - idx)
                        .take_while(|(_, a)| a.1 == first)
                        .last()
                        .expect("start_idx")
                        .0
                } else {
                    return None;
                }
            } else {
                0
            };

            let mut matches = vec![];

            state
                .first_atoms
                .iter()
                .skip(start_idx)
                .take_while(|a| rule_first_atoms.is_none() || a.1 == rule_first_atoms.unwrap())
                .for_each(|(s_i, _)| {
                    if let Some(has_var) = test_match_without_variables(input, &state[*s_i]) {
                        matches.push((*s_i, has_var));
                    }
                });

            if matches.len() == 0 {
                return None;
            }

            input_state_matches.push((i_i, matches));
        } else if input.len() == 1 && is_var_pred(&input) {
            let matches = state
                .iter()
                .enumerate()
                .map(|(i, _)| (i, true))
                .collect::<Vec<_>>();
            input_state_matches.push((i_i, matches))
        }
    }

    // precompute values required for deriving branch indices.
    let mut input_rev_permutation_counts = vec![1; input_state_matches.len()];
    let mut permutation_count = 1;
    input_state_matches
        .iter()
        .enumerate()
        .rev()
        .for_each(|(i, (_, matches))| {
            permutation_count *= matches.len();

            if i > 0 {
                input_rev_permutation_counts[i - 1] = permutation_count;
            }
        });

    let mut variables_matched: Vec<MatchLite> = vec![];

    // we'll use state as a scratchpad for other token allocations
    let len = state.len();
    state.lock_scratch();

    'outer: for p_i in 0..permutation_count {
        variables_matched.clear();
        state.reset_scratch();

        let mut states_matched_bool = vec![false; len];

        // iterate across the graph of permutations from root to leaf, where each
        // level of the tree is an input, and each branch is a match against a state.
        // TODO: improve performance by checking inputs ordered from least to most matches
        for (c_i, (i_i, matches)) in input_state_matches.iter().enumerate() {
            let branch_idx = (p_i / input_rev_permutation_counts[c_i]) % matches.len();
            let (s_i, has_var) = matches[branch_idx];

            // a previous input in this permutation has already matched the state being checked
            if states_matched_bool[s_i] {
                continue 'outer;
            } else {
                states_matched_bool[s_i] = true;
            }

            if has_var {
                // we know the structures are compatible from the earlier matching check
                if !match_variables_assuming_compatible_structure(
                    &inputs[*i_i],
                    &state,
                    s_i,
                    &mut variables_matched,
                ) {
                    continue 'outer;
                }
            }
        }

        for input in inputs.iter().filter(|input| is_backwards_pred(input)) {
            if !match_backwards_variables(input, state, &mut variables_matched) {
                continue 'outer;
            }
        }

        for input in inputs.iter().filter(|input| is_side_pred(input)) {
            if !match_side_variables(input, state, &mut variables_matched, side_input) {
                continue 'outer;
            }
        }

        for input in inputs.iter().filter(|input| is_negated_pred(input)) {
            // check negated predicates last, so that we know about all variables
            // from the backwards and side predicates
            if state
                .iter()
                .enumerate()
                .filter(|&(s_i, _)| s_i < len && !states_matched_bool[s_i])
                .any(|(s_i, _)| {
                    match_variables_with_existing(input, state, s_i, &mut variables_matched)
                })
            {
                continue 'outer;
            }
        }

        let mut forward_concrete = vec![];
        let mut outputs_concrete = vec![];

        inputs
            .iter()
            .filter(|pred| is_concrete_pred(pred))
            .for_each(|v| {
                forward_concrete.push(assign_vars(v, state, &variables_matched));
            });

        outputs.iter().for_each(|v| {
            if is_side_pred(v) {
                let pred = assign_vars(v, state, &variables_matched);

                evaluate_side_pred(&pred, side_input);
            } else {
                outputs_concrete.push(assign_vars(v, state, &variables_matched));
            }
        });

        state.unlock_scratch();

        return Some(Rule::new(r.id, forward_concrete, outputs_concrete));
    }

    state.unlock_scratch();

    None
}

fn match_backwards_variables(
    pred: &Phrase,
    state: &mut State,
    existing_matches_and_result: &mut Vec<MatchLite>,
) -> bool {
    let pred = assign_vars(pred, state, existing_matches_and_result);

    if let Some(eval_result) = evaluate_backwards_pred(&pred) {
        let s_i = state.len();
        state.push(eval_result);

        match_variables_with_existing(&pred, state, s_i, existing_matches_and_result)
    } else {
        false
    }
}

fn match_side_variables<F>(
    pred: &Phrase,
    state: &mut State,
    existing_matches_and_result: &mut Vec<MatchLite>,
    side_input: &mut F,
) -> bool
where
    F: SideInput,
{
    let pred = assign_vars(pred, state, existing_matches_and_result);

    if let Some(eval_result) = evaluate_side_pred(&pred, side_input) {
        if eval_result.len() == 0 {
            return true;
        }

        let s_i = state.len();
        state.push(eval_result);

        match_variables_with_existing(&pred, state, s_i, existing_matches_and_result)
    } else {
        false
    }
}

fn assign_vars(tokens: &Phrase, state: &State, matches: &[MatchLite]) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];

    for token in tokens {
        if is_var_token(token) {
            if let Some(m) = matches.iter().find(|m| m.atom == token.string) {
                let mut tokens = m.to_phrase(state);
                let len = tokens.len();

                if len == 1 {
                    tokens[0].is_negated = token.is_negated;
                    tokens[0].open_depth = token.open_depth;
                    tokens[len - 1].close_depth = token.close_depth;
                } else {
                    tokens[0].is_negated = token.is_negated;
                    if token.open_depth > 0 {
                        tokens[0].open_depth += token.open_depth
                    }
                    if token.close_depth > 0 {
                        tokens[len - 1].close_depth += token.close_depth
                    }
                }

                result.append(&mut tokens);

                continue;
            }
        }

        result.push(token.clone());
    }

    result
}

fn is_var_token(token: &Token) -> bool {
    token.flag == TokenFlag::Variable
}

fn is_backwards_pred(tokens: &Phrase) -> bool {
    if let TokenFlag::BackwardsPred(_) = tokens[0].flag {
        true
    } else {
        false
    }
}

fn is_side_pred(tokens: &Phrase) -> bool {
    tokens[0].flag == TokenFlag::Side
}

fn is_negated_pred(tokens: &Phrase) -> bool {
    tokens[0].is_negated
}

fn is_concrete_pred(tokens: &Phrase) -> bool {
    !is_negated_pred(tokens) && tokens[0].flag == TokenFlag::None
}

fn is_var_pred(tokens: &Vec<Token>) -> bool {
    !is_negated_pred(tokens) && tokens[0].flag == TokenFlag::Variable
}

fn evaluate_backwards_pred(tokens: &Phrase) -> Option<Vec<Token>> {
    match tokens[0].flag {
        TokenFlag::BackwardsPred(BackwardsPred::Plus) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();
            let n3 = tokens[3].as_number();

            match (n1, n2, n3) {
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_number(v1 + v2, 0, 1),
                ]),
                (Some(v1), None, Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new_number(v3 - v1, 0, 0),
                    tokens[3].clone(),
                ]),
                (None, Some(v2), Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new_number(v3 - v2, 0, 0),
                    tokens[2].clone(),
                    tokens[3].clone(),
                ]),
                (Some(v1), Some(v2), Some(v3)) if v1 + v2 == v3 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Lt) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 < v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Gt) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 > v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Lte) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 <= v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Gte) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 >= v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::ModNeg) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();
            let n3 = tokens[3].as_number();

            let mod_neg = |x: i32, n: i32| x - n * (x / n);

            match (n1, n2, n3) {
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_number(mod_neg(v1, v2), 0, 1),
                ]),
                _ => None,
            }
        }
        _ => unreachable!(),
    }
}

fn evaluate_side_pred<F>(tokens: &Phrase, side_input: &mut F) -> Option<Vec<Token>>
where
    F: SideInput,
{
    side_input(tokens)
}

fn test_match_without_variables(input_tokens: &Phrase, pred_tokens: &Phrase) -> Option<bool> {
    let mut pred_token_iter = pred_tokens.iter();

    let mut input_depth = 0;
    let mut pred_depth = 0;

    let mut has_var = false;

    for token in input_tokens {
        let pred_token = pred_token_iter.next();

        input_depth += token.open_depth;

        if let Some(pred_token) = pred_token {
            pred_depth += pred_token.open_depth;

            let is_var = is_var_token(token);

            if !is_var {
                if token.string != pred_token.string || input_depth != pred_depth {
                    return None;
                }
            } else {
                has_var = true;

                while input_depth < pred_depth {
                    if let Some(pred_token) = pred_token_iter.next() {
                        pred_depth += pred_token.open_depth;
                        pred_depth -= pred_token.close_depth;
                    } else {
                        return None;
                    }
                }
            }

            pred_depth -= pred_token.close_depth;
        } else {
            return None;
        }

        input_depth -= token.close_depth;
    }

    if pred_token_iter.next().is_some() {
        return None;
    }

    Some(has_var)
}

fn match_variables_with_existing(
    input_tokens: &Phrase,
    state: &State,
    s_i: usize,
    existing_matches_and_result: &mut Vec<MatchLite>,
) -> bool {
    if let Some(has_var) = test_match_without_variables(input_tokens, &state[s_i]) {
        if has_var {
            match_variables_assuming_compatible_structure(
                input_tokens,
                state,
                s_i,
                existing_matches_and_result,
            )
        } else {
            true
        }
    } else {
        false
    }
}

#[derive(Eq, PartialEq, Debug)]
struct MatchLite {
    atom: Atom,
    state_i: usize,
    depths: (u8, u8),
    range: (usize, usize),
}

impl MatchLite {
    fn as_slice<'a>(&self, state: &'a State) -> &'a [Token] {
        &state[self.state_i][self.range.0..self.range.1]
    }

    fn to_phrase(&self, state: &State) -> Vec<Token> {
        let mut phrase = self.as_slice(state).to_vec();

        let len = phrase.len();

        if len == 1 {
            phrase[0].open_depth = 0;
            phrase[0].close_depth = 0;
        } else {
            phrase[0].open_depth -= self.depths.0;
            phrase[len - 1].close_depth -= self.depths.1;
        }

        phrase
    }
}

fn match_variables_assuming_compatible_structure(
    input_tokens: &Phrase,
    state: &State,
    state_i: usize,
    existing_matches_and_result: &mut Vec<MatchLite>,
) -> bool {
    let pred_tokens = &state[state_i];

    assert!(test_match_without_variables(input_tokens, pred_tokens).is_some());

    let mut pred_token_i = 0;

    let mut input_depth = 0;
    let mut pred_depth = 0;

    for token in input_tokens {
        let pred_token = &pred_tokens[pred_token_i];
        pred_token_i += 1;

        input_depth += token.open_depth;
        pred_depth += pred_token.open_depth;

        let is_var = is_var_token(token);

        if is_var {
            // colect tokens to assign to the input variable
            let start_i = pred_token_i - 1;

            while input_depth < pred_depth {
                let pred_token = &pred_tokens[pred_token_i];
                pred_token_i += 1;

                pred_depth += pred_token.open_depth;
                pred_depth -= pred_token.close_depth;
            }

            let end_i = pred_token_i;

            let variable_already_matched = if let Some(ref existing_match) =
                existing_matches_and_result
                    .iter()
                    .find(|m| m.atom == token.string)
            {
                if !phrase_equal(
                    &existing_match.as_slice(state),
                    &pred_tokens[start_i..end_i],
                    existing_match.depths,
                    (token.open_depth, token.close_depth),
                ) {
                    // this match of the variable conflicted with an existing match
                    return false;
                }

                true
            } else {
                false
            };

            if !variable_already_matched {
                let m = MatchLite {
                    atom: token.string,
                    state_i,
                    depths: (token.open_depth, token.close_depth),
                    range: (start_i, end_i),
                };

                existing_matches_and_result.push(m);
            }
        }

        pred_depth -= pred_token.close_depth;
        input_depth -= token.close_depth;
    }

    true
}

#[inline]
fn phrase_equal(a: &Phrase, b: &Phrase, a_depths: (u8, u8), b_depths: (u8, u8)) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let len = b.len();

    if len == 1 {
        token_equal(&a[0], &b[0], true, None, None)
    } else {
        token_equal(
            &a[0],
            &b[0],
            false,
            Some((a_depths.0, 0)),
            Some((b_depths.0, 0)),
        ) && a
            .iter()
            .skip(1)
            .take(len - 2)
            .eq(b.iter().skip(1).take(len - 2))
            && token_equal(
                &a[len - 1],
                &b[len - 1],
                false,
                Some((0, a_depths.1)),
                Some((0, b_depths.1)),
            )
    }
}

#[inline]
fn token_equal(
    a: &Token,
    b: &Token,
    ignore_depth: bool,
    a_depth_diffs: Option<(u8, u8)>,
    b_depth_diffs: Option<(u8, u8)>,
) -> bool {
    let a_depth_diffs = a_depth_diffs.unwrap_or((0, 0));
    let b_depth_diffs = b_depth_diffs.unwrap_or((0, 0));

    a.string == b.string
        && a.is_negated == b.is_negated
        && (ignore_depth
            || (a.open_depth as i32 - a_depth_diffs.0 as i32
                == b.open_depth as i32 - b_depth_diffs.0 as i32
                && a.close_depth as i32 - a_depth_diffs.1 as i32
                    == b.close_depth as i32 - b_depth_diffs.1 as i32))
}

fn tokenize(string: &str, string_cache: &mut StringCache) -> Vec<Token> {
    let mut string = format!("({})", string);

    lazy_static! {
        static ref RE1: Regex = Regex::new(r"\(\s*(\S+|`[^`]+`)\s*\)").unwrap();
    }

    loop {
        // remove instances of brackets surrounding single atoms
        let string1 = string.clone();
        let string2 = RE1.replace_all(&string1, "$1");

        if string1 == string2 {
            break;
        } else {
            string = string2.into_owned();
        }
    }

    lazy_static! {
        static ref RE2: Regex = Regex::new(r"`(.*?)`").unwrap();
    }

    let string1 = string.clone();
    let mut strings = RE2
        .captures_iter(&string1)
        .map(|c| c.get(1).expect("string_capture").as_str());

    string = RE2.replace_all(&string, "`").to_string();

    lazy_static! {
        static ref RE3: Regex = Regex::new(r"\(|\)|\s+|[^\(\)\s]+").unwrap();
    }

    let tokens = RE3
        .find_iter(&string)
        .map(|m| m.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>();

    let mut result = vec![];

    let mut open_depth = 0;
    let mut close_depth = 0;

    for (i, token) in tokens.iter().enumerate() {
        if *token == "(" {
            open_depth += 1;
            continue;
        }

        for t in tokens.iter().skip(i + 1) {
            if *t == ")" {
                close_depth += 1;
            } else {
                break;
            }
        }

        if *token == ")" {
            continue;
        }

        if *token == "`" {
            result.push(Token::new(
                strings.next().expect("string"),
                open_depth,
                close_depth,
                string_cache,
            ));
        } else {
            result.push(Token::new(token, open_depth, close_depth, string_cache));
        }
        open_depth = 0;
        close_depth = 0;
    }

    result
}

pub fn build_phrase(phrase: &Phrase, string_cache: &StringCache) -> String {
    let mut tokens = vec![];

    for t in phrase {
        let mut string = String::new();

        if let Some(s) = t.as_str(string_cache) {
            string += s;
        } else {
            string += &t.as_number().expect("number").to_string();
        }

        tokens.push(format!(
            "{}{}{}{}",
            String::from("(").repeat(t.open_depth as usize),
            if t.is_negated {
                "!"
            } else if t.flag == TokenFlag::Side {
                "^"
            } else {
                ""
            },
            string,
            String::from(")").repeat(t.close_depth as usize)
        ));
    }

    tokens.join(" ")
}

pub fn build_state(state: &State, string_cache: &StringCache) -> String {
    state
        .iter()
        .map(|phrase_id| build_phrase(state.get(*phrase_id), string_cache))
        .collect::<Vec<_>>()
        .join("\n")
}

fn rule_to_string(rule: &Rule, string_cache: &StringCache) -> String {
    let inputs = rule
        .inputs
        .iter()
        .map(|p| build_phrase(p, string_cache))
        .collect::<Vec<_>>()
        .join(" . ");

    let outputs = rule
        .outputs
        .iter()
        .map(|p| build_phrase(p, string_cache))
        .collect::<Vec<_>>()
        .join(" . ");

    format!("{:5}: {} = {}", rule.id, inputs, outputs)
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

    fn match_variables(
        input_tokens: &Phrase,
        pred_tokens: &Phrase,
    ) -> Option<Vec<(Atom, Vec<Token>)>> {
        let mut result = vec![];

        let mut state = State::new();
        state.push(pred_tokens.to_vec());

        if match_variables_with_existing(input_tokens, &state, 0, &mut result) {
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
             asdf . #test: \n\
             at 3 4 wood = at 5 6 wood . at 7 8 wood",
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
             #update:\n\
             at X Y wood . at X Y fire = at X Y fire\n\
             () = #spread\n\
             \n\
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
                    tokenize("block-set 0 5 1", &mut string_cache),
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
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let mut state = State::from_phrases(&state);

            let result = rule_matches_state(&rule, &mut state, &mut |_: &Phrase| None);

            assert!(result.is_some());
            assert_eq!(result.unwrap(), expected);
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
                p.get(1).and_then(|t| StringCache::atom_to_number(t.string)),
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
                tokenize("+ 1 1 A", &mut string_cache),
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
        ];

        for (input, expected) in test_cases.drain(..) {
            assert_eq!(evaluate_backwards_pred(&input), expected);
        }
    }

    #[test]
    fn assign_vars_test() {
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
            let mut state = State::from_phrases(&state);
            assert_eq!(assign_vars(&tokens, &state, &matches), expected);
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
            assert_eq!(match_variables(&input_tokens, &pred_tokens), expected);
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

        let result = match_variables_with_existing(&input_tokens, &state, 0, &mut matches);

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
    fn tokenize_test() {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("t1", &mut string_cache),
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
    fn state_roundtrip_test() {
        let mut string_cache = StringCache::new();

        let mut state = State::new();
        state.push(tokenize("test 123", &mut string_cache));

        assert_eq!(state.get_all(), [tokenize("test 123", &mut string_cache)]);
    }
}
