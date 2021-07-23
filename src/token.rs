use crate::string_cache::*;

use regex::Regex;

use std::hash::Hash;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TokenFlag {
    None,
    Variable,
    Side,
    BackwardsPred(BackwardsPred),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BackwardsPred {
    Plus,
    Minus,
    Lt,
    Gt,
    Lte,
    Gte,
    ModNeg,
    Equal,
    Custom,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Token {
    pub atom: Atom,
    pub is_negated: bool,
    pub is_consuming: bool,
    pub flag: TokenFlag,
    pub open_depth: u8,
    pub close_depth: u8,
}

fn are_var_chars(mut chars: impl Iterator<Item = char>) -> bool {
    chars
        .next()
        .map(|c| c.is_ascii_uppercase())
        .unwrap_or(false)
        && chars.all(|c| c.is_numeric() || !c.is_ascii_lowercase())
}

fn is_wildcard_token(token: &str) -> bool {
    if token == "_" {
        return true;
    }

    let mut chars = token.chars();
    chars.next() == Some('_') && are_var_chars(chars)
}

impl Token {
    pub fn new(
        string: &str,
        open_depth: u8,
        close_depth: u8,
        string_cache: &mut StringCache,
    ) -> Token {
        let mut string = string;

        let is_consuming = if let Some('?') = string.chars().next() {
            string = string.get(1..).expect("get");
            false
        } else {
            true
        };

        let mut is_negated = false;
        let mut is_side = false;
        match string.chars().next() {
            Some('!') => {
                is_negated = true;
                string = string.get(1..).expect("get");
            }
            Some('^') => {
                is_side = true;
                string = string.get(1..).expect("get");
            }
            _ => {}
        }

        let is_var = are_var_chars(string.chars());

        let mut chars = string.chars();
        let first_char = chars.next();
        let second_char = string.chars().nth(1);

        let backwards_pred = match (first_char.unwrap_or(' '), second_char.unwrap_or(' ')) {
            ('<', '<') => Some(BackwardsPred::Custom),
            ('%', _) => Some(BackwardsPred::ModNeg),
            ('<', '=') => Some(BackwardsPred::Lte),
            ('>', '=') => Some(BackwardsPred::Gte),
            ('<', _) => Some(BackwardsPred::Lt),
            ('>', _) => Some(BackwardsPred::Gt),
            ('+', _) => Some(BackwardsPred::Plus),
            ('-', _) => Some(BackwardsPred::Minus),
            ('=', '=') => Some(BackwardsPred::Equal),

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
            atom,
            flag,
            is_negated,
            is_consuming,
            open_depth,
            close_depth,
        }
    }

    pub(crate) fn new_atom(atom: Atom, open_depth: u8, close_depth: u8) -> Token {
        Token {
            atom,
            flag: TokenFlag::None,
            is_negated: false,
            is_consuming: true,
            open_depth,
            close_depth,
        }
    }

    pub(crate) fn new_integer(n: i32, open_depth: u8, close_depth: u8) -> Token {
        Token {
            atom: StringCache::integer_to_atom(n),
            flag: TokenFlag::None,
            is_negated: false,
            is_consuming: true,
            open_depth,
            close_depth,
        }
    }

    pub fn as_str<'a>(&self, string_cache: &'a StringCache) -> Option<&'a str> {
        string_cache.atom_to_str(self.atom)
    }

    pub fn as_integer(&self) -> Option<i32> {
        StringCache::atom_to_integer(self.atom)
    }

    pub fn to_string(&self, string_cache: &StringCache) -> String {
        self.as_str(string_cache)
            .map(|s| s.to_string())
            .or_else(|| self.as_integer().map(|n| n.to_string()))
            .expect("to_string")
    }
}

/// Converts the provided `string` to a [Phrase].
pub fn tokenize(string: &str, string_cache: &mut StringCache) -> Vec<Token> {
    assert!(!string.is_empty());
    let mut string = string.to_string();

    // create iterator for strings surrounded by backticks
    lazy_static! {
        static ref RE1: Regex = Regex::new(r#""(.*?)""#).unwrap();
    }

    let string1 = string.clone();
    let mut strings = RE1
        .captures_iter(&string1)
        .map(|c| c.get(1).expect("string_capture").as_str());

    string = RE1.replace_all(&string, "\"").to_string();

    // split into tokens
    lazy_static! {
        static ref RE2: Regex = Regex::new(r"\(|\)|\s+|[^\(\)\s]+").unwrap();
    }

    let tokens = RE2
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

        if *token == ")" {
            continue;
        }

        for t in tokens.iter().skip(i + 1) {
            if *t == ")" {
                close_depth += 1;
            } else {
                break;
            }
        }

        if *token == "\"" {
            let atom = string_cache.str_to_atom(strings.next().expect("string"));
            result.push(Token::new_atom(atom, open_depth, close_depth));
        } else if is_wildcard_token(token) {
            let var_string = format!("WILDCARD{}", string_cache.wildcard_counter);
            string_cache.wildcard_counter += 1;

            result.push(Token::new(
                &var_string,
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

    // phrases should always contain tokens with a depth > 0. single variable phrases are an
    // exception to this rule, because they should be able to match whole state phrases.
    if !(result.len() == 1 && is_var_token(&result[0])) {
        result.first_mut().unwrap().open_depth += 1;
        result.last_mut().unwrap().close_depth += 1;
    }

    result
}

/// A sequence of [Tokens](Token) representing a phrase, usually produced using [tokenize].
///
/// The owned representation of a `Phrase` is `Vec<Token>`.
/// A `Phrase` occurs as a [State](crate::State) item or as an input or output item in a [Rule](crate::Rule).
pub type Phrase = [Token];
pub(crate) type VecPhrase = Vec<Token>;

pub trait PhraseGroup {
    fn groups(&self) -> PhraseGroupIterator<'_>;
    fn groups_at_depth(&self, depth: u8) -> PhraseGroupIterator<'_>;
    fn normalize(&self) -> Vec<Token>;
}

impl PhraseGroup for Phrase {
    fn groups(&self) -> PhraseGroupIterator<'_> {
        self.groups_at_depth(1)
    }

    fn groups_at_depth(&self, depth: u8) -> PhraseGroupIterator<'_> {
        PhraseGroupIterator {
            phrase: self,
            idx: 0,
            counter: PhraseGroupCounter::new_at_depth(depth),
        }
    }

    fn normalize(&self) -> Vec<Token> {
        let len = self.len();
        if len == 0 {
            return vec![];
        }

        let mut vec = self.to_vec();

        let mut interior_depth: i32 = 0;
        let mut min_interior_depth: i32 = 0;
        for (i, t) in vec.iter().enumerate() {
            if i == 0 {
                interior_depth -= t.close_depth as i32;
            } else if i == vec.len() - 1 {
                interior_depth += t.open_depth as i32;
            } else {
                interior_depth += t.open_depth as i32;
                interior_depth -= t.close_depth as i32;
            }

            min_interior_depth = interior_depth.min(min_interior_depth);
        }

        vec[0].open_depth = 1 + (-min_interior_depth) as u8;
        vec[len - 1].close_depth = 1 + (interior_depth - min_interior_depth) as u8;

        return vec;
    }
}

pub struct PhraseGroupIterator<'a> {
    phrase: &'a Phrase,
    idx: usize,
    counter: PhraseGroupCounter,
}

impl<'a> Iterator for PhraseGroupIterator<'a> {
    type Item = &'a Phrase;

    fn next(&mut self) -> Option<Self::Item> {
        for (i, t) in self.phrase.iter().enumerate().skip(self.idx) {
            if let Some(start_idx) = self.counter.count(t) {
                self.idx = i + 1;
                return Some(&self.phrase[start_idx..i + 1]);
            }
        }

        None
    }
}

pub(crate) struct PhraseGroupCounter {
    depth: u8,
    at_depth: u8,
    idx: usize,
    start_idx: Option<usize>,
    pub group_count: usize,
}

impl PhraseGroupCounter {
    pub(crate) fn new() -> Self {
        Self::new_at_depth(1)
    }

    fn new_at_depth(depth: u8) -> Self {
        PhraseGroupCounter {
            depth: 0,
            at_depth: depth,
            idx: 0,
            start_idx: None,
            group_count: 0,
        }
    }

    pub(crate) fn count(&mut self, token: &Token) -> Option<usize> {
        self.depth += token.open_depth;
        if self.start_idx.is_none() && self.depth >= self.at_depth {
            self.start_idx = Some(self.idx);
        }
        self.idx += 1;
        self.depth -= token.close_depth;
        if self.depth <= self.at_depth {
            if let Some(start_idx) = self.start_idx.take() {
                self.group_count += 1;
                return Some(start_idx);
            }
        }
        None
    }
}

#[inline]
pub(crate) fn token_equal(
    a: &Token,
    b: &Token,
    ignore_depth: bool,
    a_depth_diffs: Option<(u8, u8)>,
    b_depth_diffs: Option<(u8, u8)>,
) -> bool {
    let a_depth_diffs = a_depth_diffs.unwrap_or((0, 0));
    let b_depth_diffs = b_depth_diffs.unwrap_or((0, 0));

    a.atom == b.atom
        && a.is_negated == b.is_negated
        && (ignore_depth
            || (a.open_depth as i32 - a_depth_diffs.0 as i32
                == b.open_depth as i32 - b_depth_diffs.0 as i32
                && a.close_depth as i32 - a_depth_diffs.1 as i32
                    == b.close_depth as i32 - b_depth_diffs.1 as i32))
}

pub(crate) fn is_concrete_token(token: &Token) -> bool {
    token.flag == TokenFlag::None
}

pub(crate) fn is_var_token(token: &Token) -> bool {
    token.flag == TokenFlag::Variable
}

pub(crate) fn is_backwards_pred(tokens: &Phrase) -> bool {
    matches!(tokens[0].flag, TokenFlag::BackwardsPred(_))
}

pub(crate) fn is_side_pred(tokens: &Phrase) -> bool {
    tokens[0].flag == TokenFlag::Side
}

pub(crate) fn is_negated_pred(tokens: &Phrase) -> bool {
    tokens[0].is_negated
}

pub(crate) fn is_concrete_pred(tokens: &Phrase) -> bool {
    !is_negated_pred(tokens) && is_concrete_token(&tokens[0])
}

pub(crate) fn is_var_pred(tokens: &Phrase) -> bool {
    !is_negated_pred(tokens) && is_var_token(&tokens[0])
}

pub(crate) fn normalize_match_phrase(
    variable_token: &Token,
    mut match_phrase: Vec<Token>,
) -> Vec<Token> {
    let len = match_phrase.len();
    match_phrase[0].is_negated = variable_token.is_negated;
    match_phrase[0].open_depth += variable_token.open_depth;
    match_phrase[len - 1].close_depth += variable_token.close_depth;
    match_phrase
}

pub trait PhraseString {
    fn to_string(&self, string_cache: &StringCache) -> String;
}

impl PhraseString for Phrase {
    fn to_string(&self, string_cache: &StringCache) -> String {
        phrase_to_string(self, string_cache)
    }
}

pub(crate) fn phrase_to_string(phrase: &Phrase, string_cache: &StringCache) -> String {
    let mut tokens = vec![];

    for t in phrase {
        let mut string = String::new();

        if let Some(s) = t.as_str(string_cache) {
            if s.chars().any(|c| c.is_whitespace()) {
                string += &format!("\"{}\"", s);
            } else {
                string += s;
            }
        } else {
            string += &t.as_integer().expect("integer").to_string();
        }

        tokens.push(format!(
            "{}{}{}{}{}",
            String::from("(").repeat(t.open_depth as usize),
            if !t.is_consuming { "?" } else { "" },
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

pub(crate) fn test_phrase_pattern_match<const N: usize>(
    phrase: &Phrase,
    pattern: [Option<Atom>; N],
    match_pattern_length: bool,
) -> bool {
    if phrase.len() < N || (match_pattern_length && phrase.len() != N) {
        return false;
    }
    if pattern
        .iter()
        .enumerate()
        .any(|(i, atom)| atom.filter(|atom| phrase[i].atom != *atom).is_some())
    {
        return false;
    }
    true
}
